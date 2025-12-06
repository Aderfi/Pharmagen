# Pharmagen - Training Engine
# Unified Trainer Class.
# Handles Training, Validation, Metrics, and Checkpointing.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Type

import torch
import torch.nn as nn
import optuna
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm.auto import tqdm

from src.cfg.manager import DIRS
from src.modeling import DeepFM_PGenModel
from src.losses import MultiTaskUncertaintyLoss, FocalLoss, AdaptiveFocalLoss, AsymmetricLoss, PolyLoss

logger = logging.getLogger(__name__)

class PGenTrainer:
    """
    Handles the training lifecycle of the DeepFM model.
    Adheres to SRP: Only focuses on the training loop logic.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        target_cols: List[str],
        multi_label_cols: Set[str],
        params: Dict[str, Any],
        uncertainty_module: Optional[MultiTaskUncertaintyLoss] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.target_cols = target_cols
        self.ml_cols = multi_label_cols
        self.params = params
        self.uncertainty_module = uncertainty_module
        
        self.scaler = GradScaler()
        self.loss_fns = self._setup_criterions()
        self.best_loss = float("inf")
        self.patience_counter = 0

        # Ensure model directory exists
        DIRS["models"].mkdir(parents=True, exist_ok=True)

    def _criterions_choices(self) -> Dict[str, Any]:
        """Available loss functions mapped by name with their configurations."""
        # Common params
        smoothing = self.params.get("label_smoothing", 0.0)
        
        return {
            "bce": lambda: nn.BCEWithLogitsLoss(),
            "ce": lambda: nn.CrossEntropyLoss(label_smoothing=smoothing),
            "focal": lambda: FocalLoss(
                gamma=self.params.get("focal_gamma", 2.0), 
                label_smoothing=smoothing
            ),
            "adaptive_focal": lambda: AdaptiveFocalLoss(
                gamma=self.params.get("focal_gamma", 2.0), 
                label_smoothing=smoothing
            ),
            "asymmetric": lambda: AsymmetricLoss(
                gamma_neg=self.params.get("asl_gamma_neg", 4.0),
                gamma_pos=self.params.get("asl_gamma_pos", 1.0),
                clip=self.params.get("asl_clip", 0.05)
            ),
            "poly": lambda: PolyLoss(
                epsilon=self.params.get("poly_epsilon", 1.0), 
                reduction='mean', 
                label_smoothing=smoothing
            )
        }

    def _setup_criterions(self) -> Dict[str, nn.Module]:
        """Initialize loss functions per target based on configuration."""
        criterions = {}
        choices = self._criterions_choices()
        
        ml_loss_name = self.params.get("ml_loss", "bce")
        mc_loss_name = self.params.get("mc_loss", "focal")
        
        for col in self.target_cols:
            # Determine requested loss name based on column type (Multi-label vs Multi-class)
            target_loss = ml_loss_name if col in self.ml_cols else mc_loss_name
            
            # Fallback if requested loss is not in choices
            if target_loss not in choices:
                logger.warning(f"Loss '{target_loss}' not found for {col}. Using defaults.")
                target_loss = "bce" if col in self.ml_cols else "focal"
                
            criterions[col] = choices[target_loss]().to(self.device)
            
        return criterions

    def _compute_step(self, batch: Any) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes forward pass, loss, and basic accuracy metrics for a batch.
        Handles both Tabular (Dict of Tensors) and Graph (Dict with specific keys) batches.
        """
        # 1. Dispatch based on Batch Structure (Duck Typing)
        if isinstance(batch, dict) and "drug_data" in batch and "gene_input" in batch:
            return self._step_graph(batch)
        else:
            return self._step_tabular(batch)

    def _step_tabular(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Step logic for Tabular DeepFM."""
        # Filter inputs based on model feature names
        # Note: Requires model to have 'feature_names' attribute (DeepFM has it)
        feature_names = getattr(self.model, "categorical_names", getattr(self.model, "feature_names", []))
        
        inputs = {k: v.to(self.device) for k, v in batch.items() if k in feature_names}
        targets = {k: v.to(self.device) for k, v in batch.items() if k in self.target_cols}
        
        # Tabular model expects a single dict argument
        outputs = self.model(inputs)
        return self._calculate_loss_and_metrics(outputs, targets)

    def _step_graph(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Step logic for Graph Model."""
        drug_data = batch["drug_data"].to(self.device)
        gene_input = batch["gene_input"].to(self.device)
        
        targets_raw = batch["targets"]
        targets = {k: v.to(self.device) for k, v in targets_raw.items() if k in self.target_cols}
        
        # Graph model expects (drug_data, gene_input)
        outputs = self.model(drug_data, gene_input)
        return self._calculate_loss_and_metrics(outputs, targets)

    def _calculate_loss_and_metrics(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]):
        """Shared loss and metric calculation."""
        # 1. Compute Losses
        losses_per_task = {}
        for t_col, t_true in targets.items():
            pred = outputs[t_col]
            target = t_true.float() if t_col in self.ml_cols else t_true
            losses_per_task[t_col] = self.loss_fns[t_col](pred, target)
            
        # Aggregate Loss
        if self.uncertainty_module:
            total_loss = self.uncertainty_module(losses_per_task)
        else:
            total_loss = sum(losses_per_task.values())

        # 2. Compute Basic Accuracy
        accuracies = []
        with torch.no_grad():
            for t_col, t_true in targets.items():
                pred = outputs[t_col]
                if t_col in self.ml_cols:
                    probs = torch.sigmoid(pred)
                    preds_bin = (probs > 0.5).float()
                    acc = (preds_bin == t_true.float()).float().mean()
                else:
                    acc = (pred.argmax(1) == t_true).float().mean()
                accuracies.append(acc.item())
        
        avg_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        
        return total_loss, {"loss": total_loss.item(), "acc": avg_acc}

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_metrics = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)
        
        for batch in tqdm(loader, desc="Train", leave=False):
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type=self.device.type):
                loss, metrics = self._compute_step(batch)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            for k, v in metrics.items():
                total_metrics[k] += v
            
        return {k: v / n_batches for k, v in total_metrics.items()}

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_metrics = {"loss": 0.0, "acc": 0.0}
        n_batches = len(loader)
        
        with torch.inference_mode():
            for batch in loader:
                _, metrics = self._compute_step(batch)
                for k, v in metrics.items():
                    total_metrics[k] += v
                
        return {k: v / n_batches for k, v in total_metrics.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, patience: int, trial: Optional[optuna.Trial] = None) -> float:
        logger.info(f"Starting training on {self.device} for {epochs} epochs.")
        
        for epoch in range(1, epochs + 1):
            t_metrics = self.train_epoch(train_loader)
            v_metrics = self.validate(val_loader)
            
            v_loss = v_metrics["loss"]
            
            logger.info(
                f"Epoch {epoch:02d} | "
                f"Train Loss: {t_metrics['loss']:.4f} Acc: {t_metrics['acc']:.2%} | "
                f"Val Loss: {v_loss:.4f} Acc: {v_metrics['acc']:.2%}"
            )
            
            # Scheduler Step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(v_loss)
            else:
                self.scheduler.step()

            # Optuna Pruning
            if trial:
                trial.report(v_loss, epoch)
                if trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch}")
                    raise optuna.TrialPruned()

            # Early Stopping & Checkpointing
            if v_loss < self.best_loss:
                self.best_loss = v_loss
                self.patience_counter = 0
                self._save_checkpoint("best_model.pth")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break
        
        # Reload best model before returning
        self._load_checkpoint("best_model.pth")
        return self.best_loss

    def _save_checkpoint(self, name: str):
        path = DIRS["models"] / f"pmodel_{name}" if not name.startswith("pmodel_") else DIRS["models"] / name
        state = {"model": self.model.state_dict()}
        if self.uncertainty_module:
            state["uncertainty"] = self.uncertainty_module.state_dict()
        torch.save(state, path)

    def _load_checkpoint(self, name: str):
        path = DIRS["models"] / f"pmodel_{name}" if not name.startswith("pmodel_") else DIRS["models"] / name
        if path.exists():
            state = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state["model"])
            if self.uncertainty_module and "uncertainty" in state:
                self.uncertainty_module.load_state_dict(state["uncertainty"])
            logger.info(f"Loaded checkpoint from {path}")