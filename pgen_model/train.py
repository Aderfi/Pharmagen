import math
import warnings
from pathlib import Path

import torch
from src.config.config import *
from tqdm import tqdm
from typing import overload, Literal

from .model import DeepFM_PGenModel
from .model_configs import get_model_config

trained_encoders_path = Path(MODELS_DIR)

@overload
def train_model(
    train_loader,
    val_loader,
    model,
    criterions,
    epochs,
    patience,
    model_name,
    device=None,
    target_cols=None,
    scheduler=None,
    params_to_txt=None,
    multi_label_cols: set = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: dict = None,
    return_per_task_losses: Literal[True] = ...,  # <-- Literal[True]
    use_amp: bool = True,
) -> tuple[float, list[float], list[float]]: ...  # <-- Retorna 3

# Overload 2: Si return_per_task_losses=False → Retorna 2 valores
@overload
def train_model(
    train_loader,
    val_loader,
    model,
    criterions,
    epochs,
    patience,
    model_name,
    device=None,
    target_cols=None,
    scheduler=None,
    params_to_txt=None,
    multi_label_cols: set = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: dict = None,
    return_per_task_losses: Literal[False] = ...,  # <-- Literal[False]
    use_amp: bool = True,
) -> tuple[float, list[float]]: ...  # <-- Retorna 2

def train_model(
    train_loader,
    val_loader,
    model,
    criterions,  # lista: [criterions... , optimizer]
    epochs,
    patience,
    model_name, 
    device=None,
    target_cols=None,
    scheduler=None,
    params_to_txt=None,
    multi_label_cols: set = None,  # type: ignore
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: dict = None,  # type: ignore
    return_per_task_losses: bool = False,
    use_amp: bool = True,  # Enable automatic mixed precision by default
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target_cols is None:
        raise ValueError(
            "Debes definir 'target_cols' como una lista de nombres de salida"
        )

    num_targets = len(target_cols)
    
    best_loss = float("inf")
    best_accuracies = [0.0] * num_targets
    trigger_times = 0

    model = model.to(device)

    optimizer = criterions[-1]
    criterions_ = criterions[:-1]
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if (use_amp and torch.cuda.is_available()) else None

    if len(criterions_) != num_targets:
        raise ValueError(
            f"Desajuste: Se recibieron {len(criterions_)} funciones de pérdida, pero se esperaban {num_targets} (basado en 'target_cols')."
        )

    for epoch in (tqdm(range(epochs), desc="Progress", unit="epoch") if progress_bar else range(epochs)):
        model.train()
        total_loss = 0

        if progress_bar:
            train_bar = tqdm(
                train_loader,
                desc="Progreso de Época",
                unit="batch",
                colour="green",
                leave=True,
                nrows=2,
            )
            iterator = train_bar
        else:
            iterator = train_loader
        
        for batch in iterator:
            # Batch transfer to device - more efficient than individual transfers
            batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            drug = batch_device["drug"]
            genalle = batch_device["genalle"]
            gene = batch_device["gene"]
            allele = batch_device["allele"]
            targets = {col: batch_device[col] for col in target_cols}

            optimizer.zero_grad()

            # Mixed precision training context
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(drug, genalle, gene, allele)
                    
                    # Pre-allocate tensor for losses instead of using list
                    individual_losses = torch.zeros(num_targets, device=device)
                    for i, col in enumerate(target_cols):
                        loss_fn = criterions_[i]
                        pred = outputs[col]
                        true = targets[col]
                        individual_losses[i] = loss_fn(pred, true)
                    
                    # Create dictionary for weighted loss
                    unweighted_losses_dict = {
                        col: individual_losses[i] for i, col in enumerate(target_cols)
                    }

                    # Si use_weighted_loss es True, usamos la función de pérdida ponderada.
                    if use_weighted_loss:
                        loss = model.calculate_weighted_loss(unweighted_losses_dict, task_priorities)
                    else:
                        loss = individual_losses.sum()
                
                # Scaled backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without AMP
                outputs = model(drug, genalle, gene, allele)
                
                # Pre-allocate tensor for losses instead of using list
                individual_losses = torch.zeros(num_targets, device=device)
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_[i]
                    pred = outputs[col]
                    true = targets[col]
                    individual_losses[i] = loss_fn(pred, true)
                
                # Create dictionary for weighted loss
                unweighted_losses_dict = {
                    col: individual_losses[i] for i, col in enumerate(target_cols)
                }

                # Si use_weighted_loss es True, usamos la función de pérdida ponderada.
                if use_weighted_loss:
                    loss = model.calculate_weighted_loss(unweighted_losses_dict, task_priorities)
                else:
                    loss = individual_losses.sum()
                
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if progress_bar==True:
                train_bar.set_postfix({"TrainLoss": f"{loss.item():.4f}"}) # type: ignore

        ##### Validación #####
        model.eval()
        val_loss = 0
        corrects = [0] * num_targets
        totals = [0] * num_targets

        individual_loss_sums = [0.0] * num_targets

        with torch.no_grad():
            for batch in val_loader:
                # Batch transfer to device - more efficient
                batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                
                drug = batch_device["drug"]
                genalle = batch_device["genalle"]
                gene = batch_device["gene"]
                allele = batch_device["allele"]
                targets = {col: batch_device[col] for col in target_cols}

                # Use AMP for validation as well (inference only, no gradient computation)
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(drug, genalle, gene, allele)

                        # Pre-allocate tensor for losses instead of using list
                        individual_losses_val = torch.zeros(num_targets, device=device)
                        for i, col in enumerate(target_cols):
                            loss_fn = criterions_[i]
                            pred = outputs[col]
                            true = targets[col]
                            individual_losses_val[i] = loss_fn(pred, true)
                        
                        # Update individual loss sums
                        for i in range(num_targets):
                            individual_loss_sums[i] += individual_losses_val[i].item()

                        individual_losses_val_dict = {
                            col: individual_losses_val[i] for i, col in enumerate(target_cols)
                        }

                        if use_weighted_loss:
                            # Modo "Entrenamiento Final": usa la ponderación de incertidumbre
                            loss = model.calculate_weighted_loss(individual_losses_val_dict, task_priorities)
                        else:
                            # Modo "Optuna": usa la suma simple
                            loss = individual_losses_val.sum()
                else:
                    outputs = model(drug, genalle, gene, allele)

                    # Pre-allocate tensor for losses instead of using list
                    individual_losses_val = torch.zeros(num_targets, device=device)
                    for i, col in enumerate(target_cols):
                        loss_fn = criterions_[i]
                        pred = outputs[col]
                        true = targets[col]
                        individual_losses_val[i] = loss_fn(pred, true)
                    
                    # Update individual loss sums
                    for i in range(num_targets):
                        individual_loss_sums[i] += individual_losses_val[i].item()

                    individual_losses_val_dict = {
                        col: individual_losses_val[i] for i, col in enumerate(target_cols)
                    }

                    if use_weighted_loss:
                        # Modo "Entrenamiento Final": usa la ponderación de incertidumbre
                        loss = model.calculate_weighted_loss(individual_losses_val_dict, task_priorities)
                    else:
                        # Modo "Optuna": usa la suma simple
                        loss = individual_losses_val.sum()
                # --- FIN DE LA MODIFICACIÓN ---
                
                val_loss += loss.item()

                # Calcular accuracies dinámicamente
                for i, col in enumerate(target_cols):
                    pred = outputs[col]  # Logits
                    true = targets[col]  # Longs o Floats

                    if col in multi_label_cols:
                        # <--- MODIFICACIÓN: CÁLCULO DE HAMMING ACCURACY ---
                        # --- Precisión para Multi-etiqueta (usando BCE) ---
                        probs = torch.sigmoid(pred)
                        predicted = (probs > 0.5).float()
                        
                        # Contar el número total de etiquetas INDIVIDUALES correctas
                        corrects[i] += (predicted == true).sum().item()
                        # Contar el número total de etiquetas POSIBLES
                        totals[i] += true.numel() 
                        # --- FIN DE LA MODIFICACIÓN ---

                    else:
                        # --- Precisión para Etiqueta-Única (usando CrossEntropy) ---
                        _, predicted = torch.max(pred, 1)  # Argmax
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.size(0) # Total de muestras

        val_loss /= len(val_loader)

        if optuna_check_weights == True:
            # ====================== Pérdidas individuales =====================
            avg_individual_losses = [
                loss_sum / len(val_loader) for loss_sum in individual_loss_sums
            ]
            print(f"\n--- Epoch Validation Summary ---")
            print(f"Total Weighted Val Loss: {val_loss:.5f}")
            print("--- Average Individual Task Losses (Unweighted) ---")
            for i, col in enumerate(target_cols):
                print(f"Loss {target_cols[i]}: {avg_individual_losses[i]:.5f}")
            print("-------------------------------------------\n")
            exit(0)

        if scheduler is not None:
            scheduler.step(val_loss)

        # Esta lógica ahora calcula correctamente la media (Hamming o estándar)
        val_accuracies = [c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)]

        # Lógica de Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracies = val_accuracies.copy()
            trigger_times = 0

            save_model(
                model, 
                target_cols, 
                best_loss, 
                best_accuracies, 
                model_name=model_name,
                params_to_txt=params_to_txt
            )

        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping triggered after {patience} epochs. \b\r")
                break
                
    if return_per_task_losses:
        avg_per_task_losses = [loss_sum / len(val_loader) for loss_sum in individual_loss_sums] # type: ignore
        return best_loss, best_accuracies, avg_per_task_losses  # <--- Nuevo return
    else:
        return best_loss, best_accuracies  # <--- Return original

def save_model(model, target_cols, best_loss, best_accuracies, model_name, params_to_txt=None):
    try:
        model_save_dir = Path(MODELS_DIR)
    except Exception:
        model_save_dir = Path(MODELS_DIR) # Usar el directorio global
    
    # Definir archivo de pesos.
    model_save_dir.mkdir(parents=True, exist_ok=True)
    model_file_name = str("-".join([target[:3] for target in target_cols]))
    model_file_name = model_name
    
    path_model_file = model_save_dir / f"pmodel_{model_file_name}.pth"

    # Definir archivo de reporte.
    path_txt_file = model_save_dir / "txt_files"
    path_txt_file.mkdir(parents=True, exist_ok=True)
    report_filename = f"report_{model_file_name}_{round(best_loss, 5)}.txt"
    file_report = path_txt_file / report_filename

    torch.save(model.state_dict(), path_model_file)
    
    try:
        with open(file_report, "w") as f:
            f.write(f"Model Targets: {target_cols}\n")
            f.write(f"Validation Loss: {best_loss}\n")
            for i, col in enumerate(target_cols):
                f.write(f"Best Accuracy {col}: {best_accuracies[i]:.4f}\n")
            
            f.write("\nBest Parameters:\n")
            if params_to_txt:
                for key, val in params_to_txt.items():
                    f.write(f"  {key}: {val}\n")
            else:
                f.write("  No disponibles.\n")
                
    except Exception as e:
        print(f"Error al guardar el reporte .txt: {e}")
    