import torch

from pathlib import Path

from src.config.config import *
from .model_configs import MASTER_WEIGHTS

import warnings

trained_encoders_path = Path(MODELS_DIR)

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
    multi_label_cols: set = None  # type: ignore
):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if target_cols is None:
        raise ValueError("Debes definir 'target_cols' como una lista de nombres de salida")
    
    # <-- CAMBIO: Inicializa el set si es None
    if multi_label_cols is None:
        multi_label_cols = set()
        
    num_targets = len(target_cols)
    
    # ============ WEIGHTS (DINÁMICOS) en model_configs.py ============
    weights_list = []
    for col in target_cols:
        if col not in MASTER_WEIGHTS:
            warnings.warn(f"No se encontró un peso definido para el target '{col}'. Usando peso por defecto 1.0.")
            weights_list.append(1.0)
        else:
            weights_list.append(MASTER_WEIGHTS[col])
            
    weights = torch.tensor(weights_list).to(device)
    
    best_loss = float('inf')
    best_accuracies = [0.0] * num_targets
    trigger_times = 0
    model = model.to(device)

    optimizer = criterions[-1]
    criterions_ = criterions[:-1]
    
    if len(criterions_) != num_targets:
        raise ValueError(f"Desajuste: Se recibieron {len(criterions_)} funciones de pérdida, pero se esperaban {num_targets} (basado en 'target_cols').")

    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            drug = batch['drug'].to(device)
            gene = batch['gene'].to(device)
            allele = batch['allele'].to(device)
            genotype = batch['genotype'].to(device)
            
            # Los tensores de target ya vienen con el dtype correcto (long o float) desde PGenDataset
            targets = {col: batch[col].to(device) for col in target_cols}
            
            optimizer.zero_grad()
            
            outputs = model(drug, gene, allele, genotype)
            
            # El bucle de pérdida ya funciona, ya que BCEWithLogitsLoss
            # acepta (float, float) y CrossEntropyLoss acepta (float, long).
            individual_losses = []
            for i, col in enumerate(target_cols):
                loss_fn = criterions_[i]
                pred = outputs[col]
                true = targets[col]
                
                # Comprobación de tipo por si acaso (útil para debug)
                # if col in multi_label_cols and true.dtype != torch.float32:
                #     true = true.float()
                
                individual_losses.append(loss_fn(pred, true))

            individual_losses = torch.stack(individual_losses)
            
            loss = (individual_losses * weights).sum() / float(num_targets)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # avg_loss = total_loss / len(train_loader) # No se usa
        
        ##### Validación #####
        model.eval()
        val_loss = 0
        corrects = [0] * num_targets
        totals = [0] * num_targets
        
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug'].to(device)
                gene = batch['gene'].to(device)
                allele = batch['allele'].to(device)
                genotype = batch['genotype'].to(device)
                
                targets = {col: batch[col].to(device) for col in target_cols}
                
                outputs = model(drug, gene, allele, genotype)
                
                # Bucle de pérdida de validación
                individual_losses_val = []
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_[i]
                    pred = outputs[col]
                    true = targets[col]
                    # if col in multi_label_cols and true.dtype != torch.float32:
                    #     true = true.float()
                    individual_losses_val.append(loss_fn(pred, true)) #type: ignore
                
                individual_losses_val = torch.stack(individual_losses_val)
                loss = (individual_losses_val * weights).sum() / float(num_targets)
                val_loss += loss.item()

                # --- ¡CAMBIO GRANDE AQUÍ! ---
                # Calcular accuracies dinámicamente
                for i, col in enumerate(target_cols):
                    pred = outputs[col] # Logits
                    true = targets[col] # Longs o Floats

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
                        _, predicted = torch.max(pred, 1) # Argmax
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.size(0)
                # --- FIN DEL CAMBIO ---

        val_loss /= len(val_loader)
        if scheduler is not None:
            scheduler.step(val_loss)
            
        val_accuracies = [c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)]
        min_delta = 0.5

        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            best_accuracies = val_accuracies.copy()
            trigger_times = 0

            path_model_file = Path(trained_encoders_path / f'pmodel_best_model.pth')
            path_txt_file = Path(trained_encoders_path / 'txt_files')
            filename = f'pmodel_{best_loss}.txt'
            file = Path(path_txt_file / filename)
            with open(file, 'w') as f:
                f.write(f'Model Targets: {target_cols}\n')
                f.write(f'Validation Loss: {best_loss}\n')
                for i, col in enumerate(target_cols):
                    f.write(f'Best Accuracy {col}: {best_accuracies[i]}\n')
                f.write(f'Best Parameters:\n \
                        \t  {params_to_txt}\n')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break

    return best_loss, best_accuracies