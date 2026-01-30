import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import optuna
import torch
from torch import nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.cfg.manager import DIRS
from src.losses import (
    AdaptiveFocalLoss,
    AsymmetricLoss,
    FocalLoss,
    MultiTaskUncertaintyLoss,
    PolyLoss,
)

logger = logging.getLogger("Pharmagen.Trainer")

# =============================================================================
# 1. CONFIGURATION LAYER
# =============================================================================

@dataclass(frozen=True)
class TrainerConfig:
    """
    Hyperparameter configuration for the Training Loop.
    """
    # General
    n_epochs: int = 50
    patience: int = 5
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    use_amp: bool = True  # Automatic Mixed Precision
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4

    # Loss Configuration
    label_smoothing: float = 0.0
    ml_loss_type: str = "bce"    # For Multi-Label
    mc_loss_type: str = "focal"  # For Multi-Class

    # Loss Specifics
    focal_gamma: float = 2.0
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 1.0
    poly_epsilon: float = 1.0

# =============================================================================
# 2. LOSS FACTORY (Strategy Pattern)
# =============================================================================

class LossRegistry:
    """
    Central factory for instantiating loss functions based on config.
    Decouples loss logic from the Trainer.
    """
    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.smoothing = config.label_smoothing

    def get_loss(self, name: str) -> nn.Module: # noqa
        name = name.lower()
        LOSS_FUNCS = {
            "bce": [nn.BCEWithLogitsLoss, {}],
            "ce": [nn.CrossEntropyLoss,
                        {"label_smoothing": self.smoothing}],
            "focal": [FocalLoss,
                        {"gamma": self.cfg.focal_gamma, "label_smoothing": self.smoothing}],
            "adaptive_focal": [AdaptiveFocalLoss,
                        {"gamma": self.cfg.focal_gamma, "label_smoothing": self.smoothing}],
            "asymmetric": [AsymmetricLoss,
                        {"gamma_neg": self.cfg.asl_gamma_neg, "gamma_pos": self.cfg.asl_gamma_pos}],
            "poly": [PolyLoss,
                        {"epsilon": self.cfg.poly_epsilon, "label_smoothing": self.smoothing}]
        }
        if name in LOSS_FUNCS:
            cls, params = LOSS_FUNCS[name]
            return cls(**params)
        else:
            logger.warning(f"Unknown loss '{name}', falling back to BCE/CE standard.")
            return nn.BCEWithLogitsLoss()

# =============================================================================
# 3. METRICS TRACKER
# =============================================================================

class MetricTracker:
    """
    Accumulates metrics over an epoch to avoid 'online' calculation drift.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.running_loss = 0.0
        self.correct_preds = 0
        self.total_samples = 0
        self.n_batches = 0

    def update(self, loss_val: float, batch_correct: int, batch_size: int):
        self.running_loss += loss_val
        self.correct_preds += batch_correct
        self.total_samples += batch_size
        self.n_batches += 1

    def compute(self) -> dict[str, float]:
        return {
            "loss": self.running_loss / self.n_batches if self.n_batches > 0 else 0.0,
            "acc": self.correct_preds / self.total_samples if self.total_samples > 0 else 0.0
        }

# =============================================================================
# 4. ROBUST TRAINER CLASS
# =============================================================================

class PGenTrainer:
    def __init__( # noqa
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: TrainerConfig,
        target_cols: list[str],
        multi_label_cols: set[str],
        uncertainty_module: MultiTaskUncertaintyLoss | None = None
    ):
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = config
        self.target_cols = target_cols
        self.ml_cols = multi_label_cols
        self.uncertainty_module = uncertainty_module.to(config.device) if uncertainty_module else None

        # State Management
        self.scaler = GradScaler(enabled=config.use_amp)
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 1

        # Initialize Losses
        self.loss_fns = self._build_loss_functions()

    def _build_loss_functions(self) -> dict[str, nn.Module]:
        registry = LossRegistry(self.cfg)
        losses = {}
        for col in self.target_cols:
            is_multilabel = col in self.ml_cols
            loss_name = self.cfg.ml_loss_type if is_multilabel else self.cfg.mc_loss_type
            losses[col] = registry.get_loss(loss_name).to(self.cfg.device)
        return losses

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Efficiently moves batch to device using non_blocking transfer.
        """
        return {
            k: v.to(self.cfg.device, non_blocking=True)
            for k, v in batch.items()
        }

    def _compute_forward_pass(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int, int]:
        """
        Executes Forward -> Loss -> Metric Calculation atomic step.
        Returns (Loss, Correct_Predictions, Batch_Size)
        """
        batch = self._move_batch_to_device(batch)

        # 1. Forward
        # The model adheres to the dict contract: inputs -> outputs
        outputs = self.model(batch)

        # 2. Loss Calculation
        losses_per_task = {}
        for t_col, loss_fn in self.loss_fns.items():
            if t_col not in batch:
                continue # Safety skip

            pred = outputs[t_col]
            target = batch[t_col]

            # Type adjustment: Multi-label usually needs float targets for BCE
            if t_col in self.ml_cols:
                target = target.float()

            losses_per_task[t_col] = loss_fn(pred, target)

        # Uncertainty Weighting or Sum
        if self.uncertainty_module:
            total_loss = self.uncertainty_module(losses_per_task)
        else:
            total_loss = sum(losses_per_task.values())

        # 3. Quick Accuracy (For progress bar only, not final evaluation)
        # Detailed metrics (F1, AUC) should be done in a separate Evaluator class
        correct = 0
        total_items = 0

        with torch.no_grad():
            for t_col in self.target_cols:
                if t_col not in batch:
                    continue
                pred = outputs[t_col]
                target = batch[t_col]

                if t_col in self.ml_cols:
                    # Multi-label accuracy (exact match or hamming is hard, simple threshold here)
                    probs = torch.sigmoid(pred)
                    predicted = (probs > (1/2)).float()
                    correct += (predicted == target).float().sum().item()
                    total_items += target.numel() # Total elements
                else:
                    # Multi-class
                    predicted = pred.argmax(dim=1)
                    correct += (predicted == target).sum().item()
                    total_items += target.size(0)

        # Normalize "correct" to represent an average conceptual accuracy for the batch
        # This is an approximation for monitoring.
        batch_acc_count = correct / len(self.target_cols) if self.target_cols else 0

        total_loss = cast(torch.Tensor, total_loss)
        return total_loss, int(batch_acc_count), batch[list(batch.keys())[0]].size(0)

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        tracker = MetricTracker()

        loop = tqdm(loader, desc="Train", leave=False)

        for batch in loop:
            self.optimizer.zero_grad(set_to_none=True)

            # Mixed Precision Context
            with autocast(device_type="cuda" if "cuda" in self.cfg.device else "cpu", enabled=self.cfg.use_amp):
                loss, correct, size = self._compute_forward_pass(batch)

            # Scaled Backward
            self.scaler.scale(loss).backward()

            # Gradient Clipping (Important for Transformers)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            tracker.update(loss.item(), correct, size)

            # Live Update
            current_metrics = tracker.compute()
            loop.set_postfix(loss=f"{current_metrics['loss']:.4f}")

        return tracker.compute()

    def validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        tracker = MetricTracker()

        with torch.inference_mode():
            for batch in loader:
                loss, correct, size = self._compute_forward_pass(batch)
                tracker.update(loss.item(), correct, size)

        return tracker.compute()

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, trial: optuna.Trial | None = None) -> float:
        logger.info(f"Starting training on {self.cfg.device} | AMP: {self.cfg.use_amp}")

        for epoch in range(self.start_epoch, self.cfg.n_epochs + 1):
            t_metrics = self.train_epoch(train_loader)
            v_metrics = self.validate(val_loader)

            v_loss = v_metrics["loss"]

            logger.info(
                f"Epoch {epoch:02d}/{self.cfg.n_epochs} | "
                f"Train Loss: {t_metrics['loss']:.4f} | "
                f"Val Loss: {v_loss:.4f} Acc: {v_metrics['acc']:.2%}"
            )

            # Scheduler Step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

            # Checkpointing & Early Stopping
            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

            # Optuna Pruning
            if trial:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        return self.best_loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.cfg.__dict__ # Save config for reproducibility
        }
        if self.uncertainty_module:
            state["uncertainty_state"] = self.uncertainty_module.state_dict()

        filename = "model_best.pth" if is_best else f"checkpoint_ep{epoch}.pth"
        save_path = DIRS["models"] / filename
        torch.save(state, save_path)
        logger.debug(f"Checkpoint saved: {save_path}")

    def load_checkpoint(self, path: str | Path):
        """
        Resumes training state completely.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"Checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location=self.cfg.device)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if self.uncertainty_module and "uncertainty_state" in checkpoint:
            self.uncertainty_module.load_state_dict(checkpoint["uncertainty_state"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(f"Resumed training from epoch {self.start_epoch}")
