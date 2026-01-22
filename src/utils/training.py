# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
from typing import Any

import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from src.loss_functions import AdaptiveFocalLoss, FocalLoss

logger = logging.getLogger(__name__)

# --- MAPEOS ---
OPTIMIZER_MAP: dict[str, type[torch.optim.Optimizer]] = {
    "adamw": torch.optim.AdamW, "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD, "rmsprop": torch.optim.RMSprop,
}

SCHEDULER_MAP: dict[str, type[torch.optim.lr_scheduler.LRScheduler]] = {
    "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}

# --- CREACIÓN DE COMPONENTES ---

def create_optimizer(model: nn.Module, params: dict[str, Any], uncertainty_module: nn.Module | None = None) -> torch.optim.Optimizer:
    lr = params.get("learning_rate", 1e-3)
    wd = params.get("weight_decay", 1e-4)
    opt_name = params.get("optimizer_type", "adamw").lower()
    
    param_groups = [{'params': model.parameters(), 'weight_decay': wd, 'lr': lr}]
    if uncertainty_module:
        param_groups.append({'params': uncertainty_module.parameters(), 'weight_decay': 0.0, 'lr': params.get("loss_learning_rate", lr)})

    optimizer_cls = OPTIMIZER_MAP.get(opt_name, torch.optim.Adam)
    kwargs = {}
    if opt_name == "sgd":
        kwargs["momentum"] = 0.9
        
    return optimizer_cls(param_groups, **kwargs)

def create_scheduler(optimizer: torch.optim.Optimizer, params: dict[str, Any]):
    stype = params.get("scheduler_type", "plateau").lower()
    if stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=params.get("scheduler_factor", 0.5),
            patience=params.get("scheduler_patience", 3),
        )
    elif stype == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=params.get("epochs", 50), 
            eta_min=1e-6
        )
    return None

def create_task_criterions(target_cols: list[str], multi_label_cols: set[str], params: dict[str, Any], device: torch.device) -> dict[str, nn.Module]:
    criterions = {}
    gamma = params.get("focal_gamma", 2.0)
    smoothing = params.get("label_smoothing", 0.0)
    
    for col in target_cols:
        if col in multi_label_cols:
            crit = nn.BCEWithLogitsLoss()
        elif params.get("use_focal_loss", True):
            if params.get("use_adaptive_focal", False):
                crit = AdaptiveFocalLoss(gamma=gamma, label_smoothing=smoothing)
            else:
                crit = FocalLoss(gamma=gamma, label_smoothing=smoothing)
        else:
            crit = nn.CrossEntropyLoss(label_smoothing=smoothing)
        
        criterions[col] = crit.to(device)
    return criterions

# --- MÉTRICAS ---

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_multilabel: bool) -> dict[str, float]:
    if is_multilabel:
        return {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0))
        }
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "acc": float((y_pred == y_true).mean())
    }

def calculate_task_metrics(model: nn.Module, data_loader: torch.utils.data.DataLoader, 
                           feature_cols: list[str], target_cols: list[str], 
                           multi_label_cols: set[str], device: torch.device, 
                           threshold=0.5
                        ) -> dict[str, dict[str, float]]:
    model.eval()
    all_preds: dict[str, list[torch.Tensor]] = {c: [] for c in target_cols}
    all_targets: dict[str, list[torch.Tensor]] = {c: [] for c in target_cols}

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device, non_blocking=True) for k,v in batch.items() if k in feature_cols}
            outputs = model(inputs)
            
            for col in target_cols:
                if col not in batch: continue
                logits = outputs[col]
                true_v = batch[col]
                
                if col in multi_label_cols:
                    pred = (torch.sigmoid(logits) > threshold).float()
                else:
                    pred = torch.argmax(logits, dim=1)
                
                all_preds[col].append(pred.cpu())
                all_targets[col].append(true_v.cpu())

    metrics = {}
    for col in target_cols:
        if not all_preds[col]: continue
        y_p = torch.cat(all_preds[col]).numpy()
        y_t = torch.cat(all_targets[col]).numpy()
        metrics[col] = _compute_metrics(y_t, y_p, col in multi_label_cols)
        
    return metrics