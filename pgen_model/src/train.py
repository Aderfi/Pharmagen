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
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, overload

import optuna
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp.grad_scaler import GradScaler  # PyTorch 2.x standard
from torch.amp.autocast_mode import autocast

from src.config.config import MODELS_DIR, PROJECT_ROOT
from .model import DeepFM_PGenModel

logger = logging.getLogger(__name__)


@overload
def train_model(
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Any],  # [Loss1, Loss2..., Optimizer]
    epochs: int,
    patience: int,
    model_name: str,
    feature_cols: List[str],
    target_cols: List[str],
    device: torch.device,
    scheduler: Optional[Any] = None,
    multi_label_cols: Optional[set] = None,
    task_priorities: Optional[Dict[str, float]] = None,
    trial: Optional[optuna.Trial] = None,
    params_to_txt: dict | None = None,  # Compatibilidad
    return_per_task_losses: bool = True,
    progress_bar: bool = False,
    **kwargs,  # Capturar argumentos legacy
) -> Tuple[float, List[float], List[float]]: ...


@overload
def train_model(
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Any],  # [Loss1, Loss2..., Optimizer]
    epochs: int,
    patience: int,
    model_name: str,
    feature_cols: List[str],
    target_cols: List[str],
    device: torch.device,
    scheduler: Optional[Any] = None,
    multi_label_cols: Optional[set] = None,
    task_priorities: Optional[Dict[str, float]] = None,
    trial: Optional[optuna.Trial] = None,
    params_to_txt: dict | None = None,  # Compatibilidad
    return_per_task_losses: bool = False,
    progress_bar: bool = False,
    **kwargs,  # Capturar argumentos legacy
) -> Tuple[float, List[float]]: ...


def train_model(
    train_loader,
    val_loader,
    model: DeepFM_PGenModel,
    criterions: List[Any],  # [Loss1, Loss2..., Optimizer]
    epochs: int,
    patience: int,
    model_name: str,
    feature_cols: List[str],
    target_cols: List[str],
    device: torch.device,
    scheduler: Optional[Any] = None,
    multi_label_cols: Optional[set] = None,
    task_priorities: Optional[Dict[str, float]] = None,
    trial: Optional[optuna.Trial] = None,
    params_to_txt: dict | None = None,  # Compatibilidad
    return_per_task_losses: bool = False,
    progress_bar: bool = False,
    **kwargs,  # Capturar argumentos legacy
):
    optimizer = criterions[-1]
    loss_fns = {col: fn for col, fn in zip(target_cols, criterions[:-1])}
    multi_label_cols = multi_label_cols or set()

    # Inicializar Mixed Precision Scaler
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    best_val_loss = float("inf")
    best_accuracies = []
    per_task_loss_history = []
    patience_counter = 0

    epoch_iter = tqdm(range(epochs), desc="Epochs", disable=not progress_bar)

    for epoch in epoch_iter:
        # --- TRAIN ---
        model.train()
        train_loss_accum = 0.0

        train_pbar = tqdm(
            train_loader, leave=False, disable=not progress_bar, desc="Train"
        )
        for batch in train_pbar:
            # non_blocking=True acelera la transferencia si pin_memory=True en DataLoader
            inputs = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k in feature_cols
            }
            targets = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if k in target_cols
            }

            optimizer.zero_grad(set_to_none=True)  # Más eficiente que zero_grad()

            # Contexto de precisión mixta automática
            with autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(inputs)

                losses = {}
                for t_col, t_val in targets.items():
                    # Asegurar tipos correctos para pérdidas
                    if t_col in multi_label_cols:
                        # BCE espera floats
                        losses[t_col] = loss_fns[t_col](outputs[t_col], t_val.float())
                    else:
                        # CrossEntropy espera Long
                        losses[t_col] = loss_fns[t_col](outputs[t_col], t_val)

                total_loss = model.get_weighted_loss(losses, task_priorities)

            # Backpropagation escalado
            if scaler:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            train_loss_accum += total_loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_loss_accum = 0.0
        val_task_losses = {t: 0.0 for t in target_cols}
        correct_counts = {t: 0 for t in target_cols}
        total_counts = {t: 0 for t in target_cols}

        with torch.inference_mode():
            for batch in val_loader:
                inputs = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k in feature_cols
                }
                targets = {
                    k: v.to(device, non_blocking=True)
                    for k, v in batch.items()
                    if k in target_cols
                }

                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    outputs = model(inputs)
                    losses = {}
                    for t_col, t_val in targets.items():
                        if t_col in multi_label_cols:
                            loss_val = loss_fns[t_col](outputs[t_col], t_val.float())
                        else:
                            loss_val = loss_fns[t_col](outputs[t_col], t_val)

                        losses[t_col] = loss_val
                        val_task_losses[t_col] += loss_val.item()

                    total_loss = model.get_weighted_loss(losses, task_priorities)

                val_loss_accum += total_loss.item()

                # Métricas
                for t_col, t_true in targets.items():
                    pred = outputs[t_col]
                    if t_col in multi_label_cols:
                        # Hamming Accuracy (Threshold 0.5)
                        preds_bin = (torch.sigmoid(pred) > 0.5).long()
                        correct_counts[t_col] += (preds_bin == t_true).sum().item()
                        total_counts[t_col] += t_true.numel()
                    else:
                        preds_cls = torch.argmax(pred, dim=1)
                        correct_counts[t_col] += (preds_cls == t_true).sum().item()
                        total_counts[t_col] += t_true.size(0)

        avg_val_loss = val_loss_accum / len(val_loader)
        accuracies = [correct_counts[t] / max(total_counts[t], 1) for t in target_cols]
        avg_task_losses_list = [
            val_task_losses[t] / len(val_loader) for t in target_cols
        ]

        # --- UPDATES ---
        if progress_bar:
            epoch_iter.set_postfix(
                train=f"{avg_train_loss:.4f}", val=f"{avg_val_loss:.4f}"
            )

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        if trial:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_accuracies = accuracies
            per_task_loss_history = avg_task_losses_list
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping @ Epoch {epoch}")
                break

    if return_per_task_losses:
        return best_val_loss, best_accuracies, per_task_loss_history
    else:
        return best_val_loss, best_accuracies


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
        reports_dir = Path(PROJECT_ROOT) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        model_save_dir = Path(MODELS_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        path_model_file = model_save_dir / f"pmodel_{model_name}.pth"
        torch.save(model.state_dict(), path_model_file)
        logger.info(f"Model weights saved: {path_model_file}")

        # Save model
        torch.save(model, path_model_file.with_suffix(".pkl"))

        # Generate report filename
        report_filename = f"report_{model_name}_{round(best_loss, 5)}.txt"
        file_report = reports_dir / report_filename

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
