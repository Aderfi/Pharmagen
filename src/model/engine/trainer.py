# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning
# Copyright (C) 2025  Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
trainer.py

Implements the Training Logic (The "Director" in builder terms).
Handles the Training Loop, Validation, Metric Tracking, and Checkpointing.
"""

from dataclasses import asdict, dataclass
import logging
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
from src.model.metrics.losses import (
    AdaptiveFocalLoss,
    AsymmetricLoss,
    FocalLoss,
    MultiTaskUncertaintyLoss,
    PolyLoss,
)
from src.model.metrics.reporting import generate_training_report

logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONFIGURATION LAYER
# =============================================================================


@dataclass(frozen=True)
class TrainerConfig:
    """
    Hyperparameter configuration for the Training Loop.

    Acts as a parameter object for the Trainer, ensuring all settings
    (epochs, LR, device) are passed in a structured way.
    """

    # General
    n_epochs: int = 50
    patience: int = 5
    learning_rate: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Optimization
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    weight_decay: float = 1e-4

    # Loss Strategy
    label_smoothing: float = 0.0
    ml_loss_type: str = "bce"  # Multi-Label Strategy
    mc_loss_type: str = "focal"  # Multi-Class Strategy

    # Advanced Loss Params
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
    Decouples loss instantiation logic from the main Trainer loop.
    """

    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self._registry: dict[str, tuple[type[nn.Module], dict]] = {
            "bce": (nn.BCEWithLogitsLoss, {}),
            "ce": (nn.CrossEntropyLoss, {"label_smoothing": config.label_smoothing}),
            "focal": (
                FocalLoss,
                {"gamma": config.focal_gamma, "label_smoothing": config.label_smoothing},
            ),
            "adaptive_focal": (
                AdaptiveFocalLoss,
                {"gamma": config.focal_gamma, "label_smoothing": config.label_smoothing},
            ),
            "asymmetric": (
                AsymmetricLoss,
                {"gamma_neg": config.asl_gamma_neg, "gamma_pos": config.asl_gamma_pos},
            ),
            "asl": (
                AsymmetricLoss,
                {"gamma_neg": config.asl_gamma_neg, "gamma_pos": config.asl_gamma_pos},
            ),
            "poly": (
                PolyLoss,
                {"epsilon": config.poly_epsilon, "label_smoothing": config.label_smoothing},
            ),
        }

    def get_loss(self, name: str) -> nn.Module:
        """
        Instantiates a loss function by name.
        Args:
            name: 'bce', 'ce', 'focal', 'adaptive_focal', 'asymmetric'/'asl', 'poly'.
        Returns:
            Configured loss module.
        Raises:
            ValueError: Si el nombre no está registrado. Fallo ruidoso en lugar
                        de fallback silencioso — en Optuna se captura en objective()
                        y devuelve float("inf") con log del error.
        """
        name = name.lower().strip()
        if name not in self._registry:
            registered = sorted(self._registry.keys())
            raise ValueError(
                f"Loss '{name}' no registrada. Opciones disponibles: {registered}"
            )
        cls, params = self._registry[name]
        return cls(**params)


# =============================================================================
# 3. METRICS TRACKER
# =============================================================================


class MetricTracker:
    """Accumulates metrics (Loss/Accuracy) over an epoch per task."""

    def __init__(self, tasks: list[str]):
        self.tasks = tasks
        self.reset()

    def reset(self):
        self.running_loss = 0.0
        self.n_batches = 0
        self.correct_per_task = dict.fromkeys(self.tasks, 0.0)
        self.total_per_task = dict.fromkeys(self.tasks, 0)

    def update(
        self, loss_val: float, batch_corrects: dict[str, float], batch_totals: dict[str, int]
    ):
        self.running_loss += loss_val
        self.n_batches += 1
        for t in self.tasks:
            if t in batch_corrects:
                self.correct_per_task[t] += batch_corrects[t]
                self.total_per_task[t] += batch_totals[t]

    def compute(self) -> dict[str, float]:
        """Returns averaged metrics dictionary."""
        metrics = {"loss": self.running_loss / self.n_batches if self.n_batches > 0 else 0.0}

        macro_acc = 0.0
        valid_tasks = 0

        for t in self.tasks:
            total = self.total_per_task[t]
            acc = self.correct_per_task[t] / total if total > 0 else 0.0
            metrics[f"acc_{t}"] = acc
            if total > 0:
                macro_acc += acc
                valid_tasks += 1

        metrics["acc_macro"] = macro_acc / valid_tasks if valid_tasks > 0 else 0.0
        return metrics


# =============================================================================
# 4. ROBUST TRAINER CLASS
# =============================================================================


class PGenTrainer:
    """
    Main Training Engine.

    Encapsulates the training state (Model, Optimizer, Scheduler) and execution
    logic (Train Step, Validation Step, Checkpointing).

    Args:
        model (nn.Module): The Pharmagen model.
        optimizer (Optimizer): PyTorch optimizer.
        scheduler (LRScheduler): PyTorch scheduler.
        config (TrainerConfig): Hyperparameters.
        target_cols (list): List of output head names.
        multi_label_cols (set): Set of multi-label targets.
        uncertainty_module (nn.Module, optional): Kendall&Gal Loss Wrapper.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        config: TrainerConfig,
        target_cols: list[str],
        multi_label_cols: set[str],
        model_name: str = "PGenModel",
        uncertainty_module: MultiTaskUncertaintyLoss | None = None,
    ):
        self.model = model.to(config.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.cfg = config
        self.target_cols = target_cols
        self.ml_cols = multi_label_cols
        self.model_name = model_name
        self.uncertainty_module = (
            uncertainty_module.to(config.device) if uncertainty_module else None
        )

        self._device_type = "cuda" if "cuda" in config.device else "cpu"

        # State Management
        self.scaler = GradScaler(enabled=config.use_amp)
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 1
        self.history: list[dict[str, Any]] = []

        # Initialize Losses via Factory
        self.loss_fns = self._build_loss_functions()

    def _build_loss_functions(self) -> dict[str, nn.Module]:
        registry = LossRegistry(self.cfg)
        losses = {}
        for col in self.target_cols:
            is_multilabel = col in self.ml_cols
            loss_name = self.cfg.ml_loss_type if is_multilabel else self.cfg.mc_loss_type
            losses[col] = registry.get_loss(loss_name).to(self.cfg.device)
        return losses

    def _move_to_device(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(self.cfg.device, non_blocking=True) for k, v in data.items()}

    def _compute_forward_pass(
        self,
        x: dict[str, torch.Tensor],
        y: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict, dict]:
        """Atomic step: Forward -> Loss -> Metrics."""
        x = self._move_to_device(x)
        y = self._move_to_device(y)

        # 1. Forward — solo las features al modelo
        outputs = self.model(x)

        # 2. Loss Calculation
        losses_per_task: dict[str, torch.Tensor] = {}
        for t_col, loss_fn in self.loss_fns.items():
            if t_col not in y:
                continue
            pred = outputs[t_col]
            target = y[t_col].float() if t_col in self.ml_cols else y[t_col]
            losses_per_task[t_col] = loss_fn(pred, target)

        if self.uncertainty_module:
            total_loss = self.uncertainty_module(losses_per_task)
        else:
            total_loss = cast(torch.Tensor, sum(losses_per_task.values()))

        # 3. Accuracy Calculation
        batch_corrects: dict[str, float] = {}
        batch_totals: dict[str, int] = {}

        with torch.no_grad():
            for t_col in self.target_cols:
                if t_col not in y:
                    continue
                pred = outputs[t_col]
                target = y[t_col]

                if t_col in self.ml_cols:
                    probs = torch.sigmoid(pred)
                    predicted = (probs > 0.5).float()
                    batch_corrects[t_col] = (predicted == target.float()).float().sum().item()
                    batch_totals[t_col] = target.numel()
                else:
                    predicted = pred.argmax(dim=1)
                    batch_corrects[t_col] = (predicted == target).sum().item()
                    batch_totals[t_col] = target.size(0)

        return total_loss, batch_corrects, batch_totals

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        """Runs one full training epoch."""
        self.model.train()
        tracker = MetricTracker(self.target_cols)
        loop = tqdm(loader, desc="Train", leave=False)

        for x_batch, y_batch in loop:
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self._device_type, enabled=self.cfg.use_amp):
                loss, corrects, totals = self._compute_forward_pass(x_batch, y_batch)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            tracker.update(loss.item(), corrects, totals)
            metrics = tracker.compute()
            loop.set_postfix(loss=f"{metrics['loss']:.4f}", acc=f"{metrics['acc_macro']:.2%}")

        return tracker.compute()

    def validate(self, loader: DataLoader) -> dict[str, float]:
        """
        Runs validation loop (No Grad).
        """
        self.model.eval()
        tracker = MetricTracker(self.target_cols)

        with torch.inference_mode():
            for x_batch, y_batch in loader:  # FIX Bug 1: desempaquetar tupla explícitamente
                with autocast(device_type=self._device_type, enabled=self.cfg.use_amp):
                    loss, corrects, totals = self._compute_forward_pass(x_batch, y_batch)
                tracker.update(loss.item(), corrects, totals)

        return tracker.compute()

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, trial: optuna.Trial | None = None
    ) -> float:
        """
        Executes the full training lifecycle (Epochs -> Val -> Early Stopping).
        """
        logger.info("Starting training on %s | AMP: %s", self.cfg.device, self.cfg.use_amp)

        for epoch in range(self.start_epoch, self.cfg.n_epochs + 1):
            t_metrics = self.train_epoch(train_loader)
            v_metrics = self.validate(val_loader)

            v_loss = v_metrics["loss"]
            self.history.append(
                {
                    "epoch": epoch,
                    "train_loss": t_metrics["loss"],
                    "val_loss": v_loss,
                    **{f"val_{k}": v for k, v in v_metrics.items() if k.startswith("acc")},
                }
            )

            acc_str = " | ".join(
                [f"{k}: {v:.2%}" for k, v in v_metrics.items() if k.startswith("acc_")]
            )
            logger.info(
                "Epoch %02d/%d | Train Loss: %.4f | Val Loss: %.4f | %s",
                epoch,
                self.cfg.n_epochs,
                t_metrics["loss"],
                v_loss,
                acc_str,
            )

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

            # Checkpointing + Early Stopping
            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

            # Optuna Pruning Hook
            if trial:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned

        logger.info("Generating Training Report...")
        generate_training_report(self.history, DIRS["reports"], self.model_name)

        return self.best_loss

    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": asdict(self.cfg),
        }
        if self.uncertainty_module:
            state["uncertainty_state"] = self.uncertainty_module.state_dict()

        filename = "model_best.pth" if is_best else f"checkpoint_ep{epoch}.pth"
        save_path = DIRS["models"] / filename
        torch.save(state, save_path)
        logger.debug("Checkpoint saved: %s", save_path)

    def load_checkpoint(self, path: str | Path):
        """Resumes training state from file."""
        path = Path(path)
        if not path.exists():
            logger.error("Checkpoint not found: %s", path)
            return

        checkpoint = torch.load(path, map_location=self.cfg.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if self.uncertainty_module and "uncertainty_state" in checkpoint:
            self.uncertainty_module.load_state_dict(checkpoint["uncertainty_state"])

        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info("Resumed training from epoch %d", self.start_epoch)
