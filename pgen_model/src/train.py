"""
Training loop for multi-task deep learning models.

This module implements the main training loop for the DeepFM model with support for:
- Multi-task learning with per-task loss functions
- Early stopping with patience
- Optuna hyperparameter optimization with pruning
- Multi-label and single-label classification
- Uncertainty weighting for automatic task balancing
- Comprehensive metric tracking and reporting

References:
    - Multi-task learning: Ruder et al., 2017
    - Early stopping: Prechelt, 1998
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

import optuna
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from src.config.config import MODELS_DIR
from .model import DeepFM_PGenModel
from .model_configs import get_model_config

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

# Multi-label classification threshold
MULTILABEL_THRESHOLD = 0.2

# Model saving directory
TRAINED_ENCODERS_PATH = Path(MODELS_DIR)

# Accuracy calculation
EPSILON = 1e-8


# ============================================================================
# Type Definitions and Overloads
# ============================================================================


@overload
def train_model( # type: ignore
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Union[nn.Module, torch.optim.Optimizer]],
    epochs: int,
    patience: int,
    model_name: str,
    device: Optional[torch.device] = None,
    target_cols: Optional[List[str]] = None,
    scheduler: Optional[Any] = None,
    params_to_txt: Optional[Dict[str, Any]] = None,
    multi_label_cols: Optional[set] = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: Optional[Dict[str, float]] = None,
    return_per_task_losses: bool = True,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[float, List[float], List[float]]:
    """Overload: return_per_task_losses=True returns 3 values."""
    ...


@overload
def train_model( # type:ignore
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Union[nn.Module, torch.optim.Optimizer]],
    epochs: int,
    patience: int,
    model_name: str,
    device: Optional[torch.device] = None,
    target_cols: Optional[List[str]] = None,
    scheduler: Optional[Any] = None,
    params_to_txt: Optional[Dict[str, Any]] = None,
    multi_label_cols: Optional[set] = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: Optional[Dict[str, float]] = None,
    return_per_task_losses: bool = False,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[float, List[float]]:
    """Overload: return_per_task_losses=False returns 2 values."""
    ...


# ============================================================================
# Main Training Function
# ============================================================================


def train_model(
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Union[nn.Module, torch.optim.Optimizer]],
    epochs: int,
    patience: int,
    model_name: str,
    device: Optional[torch.device] = None,
    target_cols: Optional[List[str]] = None,
    scheduler: Optional[Any] = None,
    params_to_txt: Optional[Dict[str, Any]] = None,
    multi_label_cols: Optional[set] = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: Optional[Dict[str, float]] = None,
    return_per_task_losses: bool = False,
    trial: Optional[optuna.Trial] = None,
) -> Union[Tuple[float, List[float]], Tuple[float, List[float], List[float]]]:
    """
    Train a multi-task deep learning model with early stopping.
    
    Implements the main training loop with support for multi-task learning,
    early stopping, Optuna integration, and comprehensive metric tracking.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: DeepFM_PGenModel instance to train
        criterions: List of loss functions followed by optimizer
                   Format: [loss_fn_1, loss_fn_2, ..., optimizer]
        epochs: Maximum number of training epochs
        patience: Number of epochs without improvement before early stopping
        model_name: Name of the model (for logging and saving)
        device: Torch device (CPU or CUDA). If None, auto-detects.
        target_cols: List of target column names. Required.
        scheduler: Optional learning rate scheduler
        params_to_txt: Optional dict of hyperparameters to save
        multi_label_cols: Set of multi-label column names
        progress_bar: If True, show progress bars
        optuna_check_weights: If True, print per-task losses and exit
        use_weighted_loss: If True, use uncertainty weighting. If False, sum losses.
        task_priorities: Optional dict of task priority weights
        return_per_task_losses: If True, return per-task losses as 3rd value
        trial: Optional Optuna trial for pruning
    
    Returns:
        If return_per_task_losses=True:
            (best_loss, best_accuracies, per_task_losses)
        If return_per_task_losses=False:
            (best_loss, best_accuracies)
    
    Raises:
        ValueError: If target_cols not provided or criterions mismatch
        RuntimeError: If model training fails
    """
    # Validate inputs
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device auto-selected: {device}")

    if target_cols is None:
        raise ValueError("target_cols must be provided as a list of target column names")

    if not isinstance(target_cols, list) or len(target_cols) == 0:
        raise ValueError("target_cols must be a non-empty list")

    if multi_label_cols is None:
        multi_label_cols = set()

    num_targets = len(target_cols)

    # Extract optimizer and loss functions
    optimizer = criterions[-1]
    criterions_list = criterions[:-1]

    if len(criterions_list) != num_targets:
        raise ValueError(
            f"Mismatch: received {len(criterions_list)} loss functions, "
            f"but expected {num_targets} (based on target_cols)"
        )

    # Initialize tracking variables
    best_loss = float("inf")
    best_accuracies = [0.0] * num_targets
    trigger_times = 0
    individual_loss_sums = [0.0] * num_targets

    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Initialize mixed precision training (AMP) for faster training on GPU
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled for faster GPU computation")

    # Training loop
    epoch_iterator = (
        tqdm(range(epochs), desc="Training", unit="epoch")
        if progress_bar
        else range(epochs)
    )

    for epoch in epoch_iterator:
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        total_loss = 0.0

        train_iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                colour="green",
                leave=False,
            )
            if progress_bar
            else train_loader
        )

        for batch in train_iterator:
            # Move inputs to device (non_blocking for faster transfer)
            drug = batch["drug"].to(device, non_blocking=True)
            genalle = batch["genalle"].to(device, non_blocking=True)
            gene = batch["gene"].to(device, non_blocking=True)
            allele = batch["allele"].to(device, non_blocking=True)

            # Move targets to device
            targets = {col: batch[col].to(device, non_blocking=True) for col in target_cols}

            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(drug, genalle, gene, allele)
                    
                    # Calculate per-task losses
                    individual_losses = []
                    for i, col in enumerate(target_cols):
                        loss_fn = criterions_list[i]
                        pred = outputs[col]
                        true = targets[col]
                        individual_losses.append(loss_fn(pred, true)) # type: ignore
                    
                    # Combine losses
                    loss = torch.stack(individual_losses).sum()
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward() # type: ignore
                scaler.step(optimizer) # type: ignore
                scaler.update() # type: ignore
            else:
                outputs = model(drug, genalle, gene, allele)
                
                # Calculate per-task losses
                individual_losses = []
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_list[i]
                    pred = outputs[col]
                    true = targets[col]
                    individual_losses.append(loss_fn(pred, true)) # type: ignore
                
                # Combine losses
                loss = torch.stack(individual_losses).sum()
                
                # Backward pass
                loss.backward()
                optimizer.step() # type: ignore

            total_loss += loss.item()

            if progress_bar:
                train_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ====================================================================
        # Validation Phase
        # ====================================================================
        model.eval()
        val_loss = 0.0
        corrects = [0] * num_targets
        totals = [0] * num_targets
        individual_loss_sums = [0.0] * num_targets

        with torch.no_grad():
            for batch in val_loader:
                # Move inputs to device (non_blocking for faster transfer)
                drug = batch["drug"].to(device, non_blocking=True)
                genalle = batch["genalle"].to(device, non_blocking=True)
                gene = batch["gene"].to(device, non_blocking=True)
                allele = batch["allele"].to(device, non_blocking=True)

                # Move targets to device
                targets = {col: batch[col].to(device, non_blocking=True) for col in target_cols}

                # Forward pass
                outputs = model(drug, genalle, gene, allele)

                # Calculate per-task losses
                individual_losses_val = []
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_list[i]
                    pred = outputs[col]
                    true = targets[col]
                    individual_losses_val.append(loss_fn(pred, true)) # type: ignore

                # Track individual losses
                individual_losses_val_tensor = torch.stack(individual_losses_val)
                for i in range(num_targets):
                    individual_loss_sums[i] += individual_losses_val_tensor[i].item()

                # Combine losses
                individual_losses_val_dict = {
                    col: individual_losses_val[i] for i, col in enumerate(target_cols)
                }

                loss = torch.stack(individual_losses_val).sum()
                
                val_loss += loss.item()

                # Calculate per-task accuracies
                for i, col in enumerate(target_cols):
                    pred = outputs[col]
                    true = targets[col]

                    if col in multi_label_cols:
                        # Multi-label: Hamming accuracy
                        probs = torch.sigmoid(pred)
                        predicted = (probs > MULTILABEL_THRESHOLD).float()
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.numel()
                    else:
                        # Single-label: standard accuracy
                        _, predicted = torch.max(pred, 1)
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.size(0)

        # Normalize validation loss
        val_loss /= len(val_loader)

        # Optuna pruning (if applicable)
        if trial is not None:
            try:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch} (val_loss={val_loss:.4f})")
                    raise optuna.TrialPruned()
            except NotImplementedError:
                # Multi-objective optimization doesn't support pruning
                pass

        # Debug mode: print per-task losses and exit
        if optuna_check_weights:
            avg_individual_losses = [
                loss_sum / len(val_loader) for loss_sum in individual_loss_sums
            ]
            logger.info("=" * 60)
            logger.info("Epoch Validation Summary")
            logger.info("=" * 60)
            logger.info(f"Total Weighted Val Loss: {val_loss:.5f}")
            logger.info("Average Individual Task Losses (Unweighted):")
            for i, col in enumerate(target_cols):
                logger.info(f"  {col}: {avg_individual_losses[i]:.5f}")
            logger.info("=" * 60)
            raise SystemExit(0)

        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)

        # Calculate accuracies
        val_accuracies = [
            c / t if t > EPSILON else 0.0 for c, t in zip(corrects, totals)
        ]

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracies = val_accuracies.copy()
            trigger_times = 0
            logger.debug(
                f"Epoch {epoch}: New best loss {best_loss:.5f}, "
                f"accuracies {best_accuracies}"
            )
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch - patience + 1} epochs "
                    f"without improvement"
                )
                break

    # Return results based on flag
    if return_per_task_losses:
        avg_per_task_losses = [
            loss_sum / len(val_loader) for loss_sum in individual_loss_sums
        ]
        return best_loss, best_accuracies, avg_per_task_losses
    else:
        return best_loss, best_accuracies


# ============================================================================
# Model Saving Function
# ============================================================================


def save_model(
    model: DeepFM_PGenModel,
    target_cols: List[str],
    best_loss: float,
    best_accuracies: List[float],
    model_name: str,
    avg_per_task_losses: List[float],
    params_to_txt: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save trained model and generate report.
    
    Saves model weights to .pth file and generates a text report with
    training metrics and hyperparameters.
    
    Args:
        model: Trained DeepFM_PGenModel instance
        target_cols: List of target column names
        best_loss: Best validation loss achieved
        best_accuracies: List of best accuracies per task
        model_name: Name of the model
        avg_per_task_losses: List of average per-task losses
        params_to_txt: Optional dict of hyperparameters to save
    
    Raises:
        IOError: If model or report cannot be saved
    """
    try:
        # Create model directory
        model_save_dir = Path(MODELS_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        path_model_file = model_save_dir / f"pmodel_{model_name}.pth"
        torch.save(model.state_dict(), path_model_file)
        logger.info(f"Model saved: {path_model_file}")

        # Create report directory
        path_txt_file = model_save_dir / "txt_files"
        path_txt_file.mkdir(parents=True, exist_ok=True)

        # Generate report filename
        report_filename = f"report_{model_name}_{round(best_loss, 5)}.txt"
        file_report = path_txt_file / report_filename

        # Write report
        with open(file_report, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("Model Configuration:\n")
            f.write(f"  Model Name: {model_name}\n")
            f.write(f"  Target Columns: {', '.join(target_cols)}\n\n")

            f.write("Training Results:\n")
            f.write(f"  Best Validation Loss: {best_loss:.5f}\n\n")

            f.write("Per-Task Average Losses:\n")
            for i, col in enumerate(target_cols):
                f.write(f"  {col}: {avg_per_task_losses[i]:.5f}\n")
            f.write("\n")

            f.write("Per-Task Best Accuracies:\n")
            for i, col in enumerate(target_cols):
                f.write(f"  {col}: {best_accuracies[i]:.4f}\n")
            f.write("\n")

            if params_to_txt:
                f.write("Hyperparameters:\n")
                for key, val in params_to_txt.items():
                    f.write(f"  {key}: {val}\n")
            else:
                f.write("Hyperparameters: Not available\n")

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Report saved: {file_report}")

    except IOError as e:
        logger.error(f"Failed to save model or report: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model saving: {e}")
        raise
