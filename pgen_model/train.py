import math
import warnings
from pathlib import Path

import torch
from src.config.config import *
from tqdm import tqdm

from .model import DeepFM_PGenModel
from .model_configs import get_model_config

trained_encoders_path = Path(MODELS_DIR)

min_delta = 0.01

pbartch = None


def train_model(
    train_loader,
    val_loader,
    model,
    criterions,  # lista: [criterions... , optimizer]
    epochs,
    patience,
    device=None,
    target_cols=None,
    scheduler=None,
    params_to_txt=None,
    multi_label_cols: set = None,  # type: ignore
    # weights_dict: dict = None,  # type: ignore
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    model_name: str = 'unknown',
):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if target_cols is None:
        raise ValueError(
            "Debes definir 'target_cols' como una lista de nombres de salida"
        )

    num_targets = len(target_cols)
    

    #weights = torch.tensor(weights_list).to(device)

    best_loss = float("inf")
    best_accuracies = [0.0] * num_targets
    trigger_times = 0
    # min_delta = 0.01

    model = model.to(device)

    optimizer = criterions[-1]
    criterions_ = criterions[:-1]

    if len(criterions_) != num_targets:
        raise ValueError(
            f"Desajuste: Se recibieron {len(criterions_)} funciones de pérdida, pero se esperaban {num_targets} (basado en 'target_cols')."
        )

    for epoch in (tqdm(range(epochs), desc="Progress", unit="epoch") if progress_bar==True else range(epochs)):
        model.train()
        total_loss = 0

        if progress_bar==True:
            train_bar = tqdm(
                        train_loader,
                        desc="Progreso de Época",
                        unit="batch",
                        colour="green",
                        leave=True,
                        nrows=2,
                    )
    
        for batch in train_loader:
            drug = batch["drug"].to(device)
            gene = batch["gene"].to(device)
            #allele = batch["allele"].to(device)
            genotype = batch["genotype"].to(device)

            # Los tensores de target ya vienen con el dtype correcto (long o float) desde PGenDataset
            targets = {col: batch[col].to(device) for col in target_cols}

            optimizer.zero_grad()

            outputs = model(drug, gene, genotype) #allele

            

            # El bucle de pérdida ya funciona, ya que BCEWithLogitLoss
            # acepta (float, float) y CrossEntropyLoss acepta (float, long).
            individual_losses = []
            for i, col in enumerate(target_cols):
                loss_fn = criterions_[i]
                pred = outputs[col]
                true = targets[col]

                individual_losses.append(loss_fn(pred, true))
            
            # (El diccionario de 'unweighted_losses_dict' se puede crear
            # pero no es necesario para la suma simple)
            # unweighted_losses_dict = {
            #     col: individual_losses[i] for i, col in enumerate(target_cols)
            # }
            
            # --- MODIFICACIÓN OPTUNA (INICIO) ---
            # 1. Comentamos la ponderación por incertidumbre (UW) para la Fase 1
            # # Llamada única y limpia a la función de UW
            # loss = model.calculate_weighted_loss(unweighted_losses_dict)
            
            # 2. Usamos una suma simple (pesos estáticos = 1.0)
            loss = torch.stack(individual_losses).sum()
            # --- MODIFICACIÓN OPTUNA (FIN) ---

            
            # ¡¡¡IMPORTANTE!!! Estas líneas faltaban en tu código.
            # Son esenciales para que el modelo entrene.
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # train_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # avg_loss = total_loss / len(train_loader) # No se usa

        ##### Validación #####
        model.eval()
        val_loss = 0
        corrects = [0] * num_targets
        totals = [0] * num_targets

        individual_loss_sums = [0.0] * num_targets

        with torch.no_grad():
            for batch in val_loader:
                drug = batch["drug"].to(device)
                gene = batch["gene"].to(device)
                #allele = batch["allele"].to(device)
                genotype = batch["genotype"].to(device)

                targets = {col: batch[col].to(device) for col in target_cols}

                outputs = model(drug, gene, genotype)

                # Bucle de pérdida de validación
            
                individual_losses_val = []
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_[i]
                    pred = outputs[col]
                    true = targets[col]
                    # if col in multi_label_cols and true.dtype != torch.float32:
                    #     true = true.float()
                    individual_losses_val.append(loss_fn(pred, true))  # type: ignore

                
                individual_losses_val_tensor = torch.stack(individual_losses_val)
                
                for i in range(num_targets):
                    individual_loss_sums[i] += individual_losses_val_tensor[i].item()

                # (El diccionario no es necesario para la suma simple)
                # individual_losses_val_dict = {
                #     col: individual_losses_val[i] for i, col in enumerate(target_cols)
                # }

                # --- MODIFICACIÓN OPTUNA (INICIO) ---
                # 1. Comentamos la ponderación por incertidumbre (UW)
                # loss = model.calculate_weighted_loss(
                #     individual_losses_val_dict
                # )
                
                # 2. Usamos una suma simple (pesos estáticos = 1.0)
                loss = torch.stack(individual_losses_val).sum()
                # --- MODIFICACIÓN OPTUNA (FIN) ---
                
                val_loss += loss.item()

                # Calcular accuracies dinámicamente
                for i, col in enumerate(target_cols):
                    pred = outputs[col]  # Logits
                    true = targets[col]  # Longs o Floats

                    if col in multi_label_cols:
                        # --- Precisión para Multi-etiqueta (usando BCE) ---
                        # 1. Aplicar Sigmoid a los logits
                        probs = torch.sigmoid(pred)
                        # 2. Obtener predicciones (umbral de 0.5)
                        predicted = (probs > 0.5).float()
                        # 3. Calcular "Exact Match Ratio"
                        # Comprueba si *todo* el vector de predicción es idéntico al vector real
                        corrects[i] += (predicted == true).all(dim=1).sum().item()
                        totals[i] += true.size(0)

                    else:
                        # --- Precisión para Etiqueta-Única (usando CrossEntropy) ---
                        _, predicted = torch.max(pred, 1)  # Argmax
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.size(0)
                # --- FIN DEL CAMBIO ---

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

            """
            """
            with open("comprobacion.txt", "a") as f:
                f.write("--- Epoch Validation Summary ---\n")
                f.write(f"Total Weighted Val Loss: {val_loss:.5f}\n")
                f.write("--- Average Individual Task Losses (Unweighted) ---\n")
                for i, col in enumerate(target_cols):
                    f.write(f"Loss {target_cols[i]}: {avg_individual_losses[i]:.5f}\n")
                f.write("-------------------------------------------\n")
            # ==================================================================
            exit(0)

        if scheduler is not None:
            scheduler.step(val_loss)

        val_accuracies = [c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)]

        if val_loss < best_loss:
            # print(f"Epoch {epoch+1}: Validation loss improved from {best_loss:.5f} to {val_loss:.5f}. Saving model.")
            best_loss = val_loss
            best_accuracies = val_accuracies.copy()
            trigger_times = 0

            save_model(model, target_cols, best_loss, best_accuracies, model_name=model_name)

        else:
            trigger_times += 1
            # print(f"Epoch {epoch+1}: Validation loss did not improve. Patience {trigger_times}/{patience}.")
            if trigger_times >= patience:
                print(f"Early stopping triggered after {patience} epochs.")
                break
    return best_loss, best_accuracies

def save_model(model, target_cols, best_loss, best_accuracies, model_name, params_to_txt=None):
    model_dir = get_model_config(model_name=model_name, path=True)
    model_file_name = str("-".join([target[:3] for target in target_cols]))
    path_model_file = Path(
                            trained_encoders_path / f"pmodel_{model_file_name}.pth"
                            )
    path_txt_file = Path(trained_encoders_path / "txt_files")
    filename = f"pmodel_{round(best_loss, 7)}.txt"
    
    
    torch.save(model.state_dict(), path_model_file)
    
    
    filename = f"pmodel_{round(best_loss, 5)}.txt"
    file = Path(path_txt_file / filename)
            
    with open(file, "w") as f:
                f.write(f"Model Targets: {target_cols}\n")
                f.write(f"Validation Loss: {best_loss}\n")
                for i, col in enumerate(target_cols):
                    f.write(f"Best Accuracy {col}: {best_accuracies[i]}\n")
                f.write(
                    f"Best Parameters:\n \
                        \t  {params_to_txt}\n"
                )

