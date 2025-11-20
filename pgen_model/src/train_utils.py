import logging
from typing import Dict, List, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from .data import PGenDataProcess
from .model_configs import MULTI_LABEL_COLUMN_NAMES
from .focal_loss import create_focal_loss


logger = logging.getLogger(__name__)


def get_input_dims(data_loader: PGenDataProcess) -> Dict[str, int]:
    """Devuelve el tamaño del vocabulario para cada columna de entrada."""
    dims = {}
    for col, encoder in data_loader.encoders.items():
        # Solo nos importan features, no targets (aunque estén en data_loader)
        if hasattr(encoder, "classes_"):
            dims[col] = len(encoder.classes_)
        else: # MultiLabel
            dims[col] = len(encoder.classes_)
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
    feature_cols: List[str],
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
            features = {col: batch[col].to(device, non_blocking=True) for col in feature_cols}
            targets = {col: batch[col].to(device, non_blocking=True) for col in target_cols}

            # Forward pass
            outputs = model(**features)

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

def create_optimizer(model: nn.Module, params: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = params.get("learning_rate", 1e-3)
    wd = params.get("weight_decay", 1e-4)
    opt_type = params.get("optimizer_type", "adamw").lower()

    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

def create_scheduler(optimizer: torch.optim.Optimizer, params: Dict[str, Any]):
    stype = params.get("scheduler_type", "plateau")
    if stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=params.get("scheduler_factor", 0.5), 
            patience=params.get("scheduler_patience", 5)
        )
    return None

def create_criterions(target_cols: List[str], params: Dict[str, Any], device: torch.device) -> List[nn.Module]:
    """Crea una lista de funciones de pérdida correspondiente al orden de target_cols."""
    criterions = []
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        elif col == "effect_type":
            # Ejemplo de uso de Focal Loss para una columna específica
            criterions.append(create_focal_loss(gamma=params.get("focal_gamma", 2.0)))
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.1)))
    return criterions