# Copyright (C) 2023 [Tu Nombre / Pharmagen Team]
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
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Set, Union, overload

import optuna
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from src.config.config import MODELS_DIR, PROJECT_ROOT
from .model import DeepFM_PGenModel

logger = logging.getLogger(__name__)

# ==============================================================================
# 1. FUNCIONES AUXILIARES (PRIVADAS)
# ==============================================================================

def _move_batch_to_device(
    batch: Dict[str, Any], 
    cols: List[str], 
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Mueve selectivamente columnas del batch al dispositivo."""
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
        if k in cols
    }

def _compute_total_loss(
    losses: Dict[str, torch.Tensor],
    priorities: Optional[Dict[str, float]],
    uncertainty_module: Optional[nn.Module] = None
) -> torch.Tensor:
    """
    Calcula la pérdida total. 
    Soporta:
    1. Ponderación automática (MultiTaskUncertaintyLoss) si se pasa el módulo.
    2. Ponderación manual (priorities).
    """
    # Estrategia 1: Incertidumbre Automática (Kendall & Gal)
    if uncertainty_module is not None:
        return uncertainty_module(losses)
    
    # Estrategia 2: Suma Ponderada Manual
    total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
    for task, loss in losses.items():
        weight = priorities.get(task, 1.0) if priorities else 1.0
        total_loss += loss * weight
    
    return total_loss

def _run_epoch(
    model: DeepFM_PGenModel,
    loader: Any,
    feature_cols: List[str],
    target_cols: List[str],
    loss_fns: Dict[str, nn.Module],
    device: torch.device,
    performance_monitor = None
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    multi_label_cols: Set[str] = set(),
    task_priorities: Optional[Dict[str, float]] = None,
    uncertainty_module: Optional[nn.Module] = None,
    is_train: bool = True,
    progress_bar: bool = False,
    desc: str = ""
) -> Tuple[float, Dict[str, float], Dict[str, int], Dict[str, int]]:
    """
    Ejecuta una época completa (entrenamiento o validación).
    Retorna: (loss_promedio, dict_losses_por_tarea, correct_counts, total_counts)
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss_accum = 0.0
    task_loss_accum = {t: 0.0 for t in target_cols}
    
    # Métricas simples para monitoreo durante validación
    correct_counts = {t: 0 for t in target_cols}
    total_counts = {t: 0 for t in target_cols}

    # Barra de progreso interna (solo si se solicita explícitamente)
    pbar = tqdm(loader, desc=desc, leave=False, disable=not progress_bar)

    # Contexto: train habilita gradientes, eval los deshabilita
    context = torch.enable_grad() if is_train else torch.inference_mode()

    with context:
        for batch in pbar:
            if performance_monitor: 
                performance_monitor.start_batch()
                
            inputs = _move_batch_to_device(batch, feature_cols, device)
            targets = _move_batch_to_device(batch, target_cols, device)

            if is_train and optimizer:
                optimizer.zero_grad(set_to_none=True)

            if performance_monitor: 
                performance_monitor.record_data_loading()

            # --- Forward Pass con Mixed Precision ---
            with autocast(device_type=device.type, enabled=(scaler is not None)):
                outputs = model(inputs)
                
                # Cálculo de pérdidas individuales
                losses = {}
                for t_col, t_true in targets.items():
                    pred = outputs[t_col]
                    loss_fn = loss_fns[t_col]
                    
                    # Asegurar tipo float para BCE
                    if t_col in multi_label_cols:
                        loss = loss_fn(pred, t_true.float())
                    else:
                        loss = loss_fn(pred, t_true)
                    
                    losses[t_col] = loss
                    task_loss_accum[t_col] += loss.item()

                # Cálculo de pérdida total
                total_loss = _compute_total_loss(losses, task_priorities, uncertainty_module)

            # --- Backward Pass (Solo Train) ---
            if is_train and optimizer:
                if scaler:
                    scaler.scale(total_loss).backward()
                    scaler.scale(total_loss) # Unscale for clipping (opcional)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

            total_loss_accum += total_loss.item()

            # --- Métricas 'on-the-fly' (Solo Val para ahorrar CPU en Train) ---
            if not is_train:
                for t_col, t_true in targets.items():
                    pred = outputs[t_col]
                    if t_col in multi_label_cols:
                        # Hamming simple (Threshold 0.5)
                        pred_bin = (torch.sigmoid(pred) > 0.5).long()
                        correct_counts[t_col] += (pred_bin == t_true).sum().item()
                        total_counts[t_col] += t_true.numel() # Total elementos (Batch * N_Labels)
                    else:
                        pred_cls = torch.argmax(pred, dim=1)
                        correct_counts[t_col] += (pred_cls == t_true).sum().item()
                        total_counts[t_col] += t_true.size(0)

    avg_loss = total_loss_accum / len(loader)
    avg_task_losses = {t: v / len(loader) for t, v in task_loss_accum.items()}

    return avg_loss, avg_task_losses, correct_counts, total_counts


# ==============================================================================
# 2. FUNCIÓN PRINCIPAL: TRAIN_MODEL
# ==============================================================================

@overload
def train_model(
    train_loader: Any, val_loader: Any, model: DeepFM_PGenModel, criterions: List[Any],
    epochs: int, patience: int, model_name: str, feature_cols: List[str], target_cols: List[str],
    device: torch.device, scheduler: Optional[Any] = None, multi_label_cols: Optional[set] = None,
    task_priorities: Optional[Dict[str, float]] = None, trial: Optional[optuna.Trial] = None,
    params_to_txt: dict | None = None, return_per_task_losses: bool = True,
    progress_bar: bool = False, **kwargs
) -> Tuple[float, List[float], List[float]]: ...

@overload
def train_model(
    train_loader: Any, val_loader: Any, model: DeepFM_PGenModel, criterions: List[Any],
    epochs: int, patience: int, model_name: str, feature_cols: List[str], target_cols: List[str],
    device: torch.device, scheduler: Optional[Any] = None, multi_label_cols: Optional[set] = None,
    task_priorities: Optional[Dict[str, float]] = None, trial: Optional[optuna.Trial] = None,
    params_to_txt: dict | None = None, return_per_task_losses: bool = False,
    progress_bar: bool = False, **kwargs
) -> Tuple[float, List[float]]: ...

def train_model(
    train_loader,
    val_loader,
    model: DeepFM_PGenModel,
    criterions: List[Any],
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
    params_to_txt: dict | None = None,
    return_per_task_losses: bool = False,
    progress_bar: bool = False,
    **kwargs,
):
    # 1. Desempaquetado Seguro
    optimizer = criterions[-1]
    
    if len(criterions) - 1 != len(target_cols):
        raise ValueError("El número de loss functions no coincide con target_cols")
    loss_fns = {col: fn for col, fn in zip(target_cols, criterions[:-1])}
    
    multi_label_cols = multi_label_cols or set()
    
    # Capturar módulo de incertidumbre si viene en kwargs (Integración con paso anterior)
    uncertainty_module = kwargs.get("uncertainty_loss_module", None)

    # 2. Configuración Inicial
    scaler = GradScaler("cuda") if device.type == "cuda" else None
    
    best_val_loss = float("inf")
    best_accuracies = []
    per_task_loss_history = []
    patience_counter = 0

    # Barra global de Epochs (Solo visible si progress_bar=True)
    epoch_iter = tqdm(range(epochs), desc="Epochs", disable=not progress_bar, file=sys.stdout)

    for epoch in epoch_iter:
        
        # --- TRAIN STEP ---
        train_loss, _, _, _ = _run_epoch(
            model=model,
            loader=train_loader,
            feature_cols=feature_cols,
            target_cols=target_cols,
            loss_fns=loss_fns,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            multi_label_cols=multi_label_cols,
            task_priorities=task_priorities,
            uncertainty_module=uncertainty_module,
            is_train=True,
            progress_bar=progress_bar, # Barra anidada
            desc="Train"
        )

        # --- VALIDATION STEP ---
        val_loss, val_task_losses, correct_counts, total_counts = _run_epoch(
            model=model,
            loader=val_loader,
            feature_cols=feature_cols,
            target_cols=target_cols,
            loss_fns=loss_fns,
            device=device,
            optimizer=None, # No optimizer en val
            scaler=scaler,
            multi_label_cols=multi_label_cols,
            task_priorities=task_priorities,
            uncertainty_module=uncertainty_module,
            is_train=False,
            progress_bar=False, # Validacion suele ser rapida, evitamos ruido visual
            desc="Val"
        )

        # --- MÉTRICAS Y LOGGING ---
        # Calculamos accuracy simple para reportar
        current_accuracies = [
            correct_counts[t] / max(total_counts[t], 1) 
            for t in target_cols
        ]
        
        # Actualizar descripción de barra de progreso
        if progress_bar:
            epoch_iter.set_postfix(
                train=f"{train_loss:.4f}", 
                val=f"{val_loss:.4f}"
            )

        # --- SCHEDULER ---
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # --- OPTUNA PRUNING ---
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                # Importante: No hacemos break, lanzamos la excepción para que Optuna la capture
                raise optuna.TrialPruned()

        # --- EARLY STOPPING & CHECKPOINT ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_accuracies = current_accuracies
            per_task_loss_history = [val_task_losses[t] for t in target_cols]
            patience_counter = 0
            # Punto de control.

            ckpt_path = Path(MODELS_DIR) / f"temp_best_{model_name}.pth"
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if progress_bar:
                    logger.info(f"Early stopping activado en Epoch {epoch}")
                break
    
    ckpt_path = Path(MODELS_DIR) / f"temp_best_{model_name}.pth"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path))

    # Retorno condicional
    if return_per_task_losses:
        return best_val_loss, best_accuracies, per_task_loss_history
    return best_val_loss, best_accuracies


# ==============================================================================
# 3. FUNCIÓN SAVE_MODEL (Optimizada con Pathlib)
# ==============================================================================

def save_model(
    model: DeepFM_PGenModel,
    target_cols: List[str],
    best_loss: float,
    best_accuracies: List[float],
    model_name: str,
    avg_per_task_losses: List[float],
    params_to_txt: Optional[Dict[str, Any]] = None,
) -> None:
    """Guarda el modelo (state_dict y pickle completo) y genera un reporte."""
    try:
        reports_dir = Path(PROJECT_ROOT) / "reports"
        model_save_dir = Path(MODELS_DIR)
        
        reports_dir.mkdir(parents=True, exist_ok=True)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # 1. Guardar Pesos (Recomendado)
        path_weights = model_save_dir / f"pmodel_{model_name}.pth"
        torch.save(model.state_dict(), path_weights)
        
        # 2. Guardar Objeto Completo (Legacy/Convenience)
        # Nota: Esto puede dar problemas si cambia la estructura de la clase
        path_full = path_weights.with_suffix(".pkl")
        torch.save(model, path_full)

        logger.info(f"Modelo guardado en: {path_weights}")

        # 3. Generar Reporte
        report_file = reports_dir / f"report_{model_name}_{datetime.now().strftime('%Y_%m_%d')}.txt"
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"{'='*70}\nMODEL TRAINING REPORT\n{'='*70}\n\n")
            f.write(f"Model Name: {model_name}\n")
            f.write(f"Validation Loss: {best_loss:.6f}\n\n")
            
            f.write("Per-Task Performance:\n")
            for col, acc, loss in zip(target_cols, best_accuracies, avg_per_task_losses):
                f.write(f"  {col:<20} | Acc: {acc:.4f} | Loss: {loss:.4f}\n")
            
            f.write("\nHyperparameters:\n")
            if params_to_txt:
                for k, v in params_to_txt.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write("  Not available\n")

    except Exception as e:
        logger.error(f"Error crítico guardando el modelo: {e}")
        raise