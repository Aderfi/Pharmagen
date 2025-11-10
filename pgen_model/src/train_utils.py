import logging

from functools import partial
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader

from .data import PGenDataProcess
from .model_configs import MULTI_LABEL_COLUMN_NAMES


logger = logging.getLogger(__name__)


def get_input_dims(data_loader: PGenDataProcess) -> Dict[str, int]:
    """
    Get vocabulary sizes for all input columns.
    
    Args:
        data_loader: Fitted PGenDataProcess instance with encoders
    
    Returns:
        Dictionary mapping input column names to vocabulary sizes
    
    Raises:
        KeyError: If required encoder not found
        AttributeError: If encoder missing 'classes_' attribute
    """
    input_cols = ["drug", "genalle", "gene", "allele"]
    dims = {}
    
    for col in input_cols:
        try:
            if col not in data_loader.encoders:
                raise KeyError(f"Encoder not found for input column: {col}")
            
            encoder = data_loader.encoders[col]
            if not hasattr(encoder, "classes_"):
                raise AttributeError(f"Encoder for '{col}' missing 'classes_' attribute")
            
            dims[col] = len(encoder.classes_)
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting input dimension for '{col}': {e}")
            raise

    return dims


def get_output_sizes(
    data_loader: PGenDataProcess, target_cols: List[str]
) -> List[int]:
    """
    Get vocabulary sizes for all target columns.
    
    Args:
        data_loader: Fitted PGenDataProcess instance with encoders
        target_cols: List of target column names
    
    Returns:
        List of vocabulary sizes corresponding to target_cols
    
    Raises:
        KeyError: If target encoder not found
    """
    sizes = []
    for col in target_cols:
        try:
            if col not in data_loader.encoders:
                raise KeyError(f"Encoder not found for target column: {col}")
            
            encoder = data_loader.encoders[col]
            if not hasattr(encoder, "classes_"):
                raise AttributeError(f"Encoder for '{col}' missing 'classes_' attribute")
            
            sizes.append(len(encoder.classes_))
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting output size for '{col}': {e}")
            raise

    return sizes


def calculate_task_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    target_cols: List[str],
    multi_label_cols: set,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate detailed metrics (F1, precision, recall) for each task.
    
    Handles both single-label and multi-label classification tasks.
    For single-label: uses argmax for predictions.
    For multi-label: uses sigmoid with 0.5 threshold.
    
    Args:
        model: Trained model in eval mode
        data_loader: DataLoader with validation/test data
        target_cols: List of target column names
        multi_label_cols: Set of multi-label column names
        device: Torch device (CPU or CUDA)
    
    Returns:
        Dictionary with structure:
        {
            'task_name': {
                'f1_macro': float,
                'f1_weighted': float,
                'precision_macro': float,
                'recall_macro': float
            }
        }
    """
    model.eval()

    # Store predictions and ground truth
    all_preds = {col: [] for col in target_cols}
    all_targets = {col: [] for col in target_cols}

    with torch.no_grad():
        for batch in data_loader:
            drug = batch["drug"].to(device)
            genalle = batch["genalle"].to(device)
            gene = batch["gene"].to(device)
            allele = batch["allele"].to(device)

            outputs = model(drug, genalle, gene, allele)

            for col in target_cols:
                true = batch[col].to(device)
                pred = outputs[col]

                if col in multi_label_cols:
                    # Multi-label: apply sigmoid and threshold
                    probs = torch.sigmoid(pred)
                    predicted = (probs > 0.5).float()
                else:
                    # Single-label: argmax
                    predicted = torch.argmax(pred, dim=1)

                all_preds[col].append(predicted.cpu())
                all_targets[col].append(true.cpu())

    # Calculate metrics per task
    metrics = {}
    for col in target_cols:
        preds = torch.cat(all_preds[col]).numpy()
        targets = torch.cat(all_targets[col]).numpy()

        if col in multi_label_cols:
            # Multi-label: use 'samples' average
            metrics[col] = {
                "f1_samples": f1_score(targets, preds, average="samples", zero_division=0),
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
                "precision_samples": precision_score(
                    targets, preds, average="samples", zero_division=0
                ),
                "recall_samples": recall_score(
                    targets, preds, average="samples", zero_division=0
                ),
            }
        else:
            # Single-label
            metrics[col] = {
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
                "precision_macro": precision_score(
                    targets, preds, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    targets, preds, average="macro", zero_division=0
                ),
            }

    return metrics


def create_optimizer(model, params):
    """Create optimizer based on params configuration."""
    opt_type = params.get("optimizer_type", "adamw")
    lr = params["learning_rate"]
    weight_decay = params.get("weight_decay", 1e-5)
    
    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(params.get("adam_beta1", 0.9), params.get("adam_beta2", 0.999))
        )
    elif opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(params.get("adam_beta1", 0.9), params.get("adam_beta2", 0.999))
        )
    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            momentum=params.get("sgd_momentum", 0.9)
        )
    elif opt_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Fallback to AdamW
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_criterions(target_cols, params, class_weights_task3=None, device=torch.device("cuda")):
    """Create loss functions based on params configuration."""
    criterions_list = []
    
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions_list.append(nn.BCEWithLogitsLoss())
        elif col == "effect_type" and class_weights_task3 is not None:
            from .focal_loss import FocalLoss
            
            gamma = params.get("focal_gamma", 2.0)
            alpha_weight = params.get("focal_alpha_weight", 1.0)
            label_smoothing = params.get("label_smoothing", 0.15)
            
            # Scale class weights by alpha_weight
            scaled_weights = class_weights_task3 * alpha_weight
            
            criterions_list.append(FocalLoss(
                alpha=scaled_weights,
                gamma=gamma,
                label_smoothing=label_smoothing,
            ))
        else:
            label_smoothing = params.get("label_smoothing", 0.1)
            criterions_list.append(nn.CrossEntropyLoss(label_smoothing=label_smoothing))
    
    return criterions_list


def create_scheduler(optimizer, params):
    """Create learning rate scheduler based on params configuration."""
    scheduler_type = params.get("scheduler_type", "plateau")
    
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params.get("scheduler_factor", 0.5),
            patience=params.get("scheduler_patience", 5),
            
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("epochs", 100),
            eta_min=params.get("learning_rate", 1e-4) * 0.01
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("scheduler_patience", 10),
            gamma=params.get("scheduler_factor", 0.5)
        )
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params.get("scheduler_factor", 0.95)
        )
    elif scheduler_type == "none":
        return None
    else:
        return None
