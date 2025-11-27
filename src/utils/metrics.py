# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani

from typing import Dict, List, Set
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

class TaskEvaluator:
    """
    Handles the evaluation logic for multi-task and multi-label models.
    Encapsulación de la lógica de calculo de métricas.
    """
    def __init__(self, device: torch.device):
        self.device = device

    def _compute_single_task_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, is_multilabel: bool) -> Dict[str, float]:
        """
        Internal method to compute metrics based on task type (SRP/KISS).
        """
        if is_multilabel:
            return {
                "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0))
            }
        
        return {
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "acc": float((y_pred == y_true).mean())
        }

    def evaluate(
        self, 
        model: nn.Module, 
        data_loader: torch.utils.data.DataLoader, 
        feature_cols: List[str], 
        target_cols: List[str], 
        multi_label_cols: Set[str], 
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Runs the evaluation loop over the dataloader.
        """
        model.eval()
        
        # Container for batch results
        all_preds: Dict[str, List[torch.Tensor]] = {c: [] for c in target_cols}
        all_targets: Dict[str, List[torch.Tensor]] = {c: [] for c in target_cols}

        with torch.no_grad():
            for batch in data_loader:
                # Prepare inputs (Dependency Inversion: Logic depends on feature_cols abstraction)
                inputs = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if k in feature_cols}
                outputs = model(inputs)
                
                for col in target_cols:
                    if col not in batch:
                        continue
                        
                    logits = outputs[col]
                    true_v = batch[col]
                    
                    if col in multi_label_cols:
                        pred = (torch.sigmoid(logits) > threshold).float()
                    else:
                        pred = torch.argmax(logits, dim=1)
                    
                    all_preds[col].append(pred.cpu())
                    all_targets[col].append(true_v.cpu())

        # Final aggregation
        metrics = {}
        for col in target_cols:
            if not all_preds[col]:
                continue
                
            y_p = torch.cat(all_preds[col]).numpy()
            y_t = torch.cat(all_targets[col]).numpy()
            metrics[col] = self._compute_single_task_metrics(y_t, y_p, col in multi_label_cols)
            
        return metrics