import torch
from pathlib import Path
from src.config.config import *


trained_encoders_path = Path(MODELS_DIR)

def train_model(
    train_loader,
    val_loader,
    model,
    criterions,  # lista: [outcome_criterion, variation_criterion, effect_criterion]
    epochs,
    patience,
    device=None,
    target_cols=None,
    scheduler=None,
    params_to_txt=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if target_cols is None or len(target_cols) != 3:
        raise ValueError("Debes definir target_cols como lista de tres nombres de salida (['outcome', 'variation', 'effect'])")
    best_loss = float('inf')
    best_accuracies = [0.0, 0.0, 0.0]  # Para guardar la mejor accuracy de cada target
    trigger_times = 0
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            drug = batch['drug'].to(device)
            gene = batch['gene'].to(device)
            allele = batch['allele'].to(device)
            genotype = batch['genotype'].to(device)
            out_target = batch[target_cols[0]].to(device)
            var_target = batch[target_cols[1]].to(device)
            eff_target = batch[target_cols[2]].to(device)
            optimizer = criterions[-1]  # optimizer is always last
            criterions_ = criterions[:-1]  # three losses
            optimizer.zero_grad()
            out_pred, var_pred, eff_pred = model(drug, gene, allele, genotype)
            loss = (
                criterions_[0](out_pred, out_target)
                + criterions_[1](var_pred, var_target)
                + criterions_[2](eff_pred, eff_target)
            ) / 3.0
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        ##### Validación #####
        
        model.eval()
        val_loss = 0
        corrects = [0, 0, 0]
        totals = [0, 0, 0]
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug'].to(device)
                gene = batch['gene'].to(device)
                allele = batch['allele'].to(device)
                genotype = batch['genotype'].to(device)
                out_target = batch[target_cols[0]].to(device)
                var_target = batch[target_cols[1]].to(device)
                eff_target = batch[target_cols[2]].to(device)
                out_pred, var_pred, eff_pred = model(drug, gene, allele, genotype)
                loss = ( 
                    criterions_[0](out_pred, out_target)    #type: ignore
                    + criterions_[1](var_pred, var_target)  #type: ignore
                    + criterions_[2](eff_pred, eff_target)  #type: ignore
                ) / 3.0
                val_loss += loss.item()
                for i, (pred, target) in enumerate([
                    (out_pred, out_target),
                    (var_pred, var_target),
                    (eff_pred, eff_target),
                ]):
                    _, predicted = torch.max(pred, 1)
                    corrects[i] += (predicted == target).sum().item()
                    totals[i] += target.size(0)
        val_loss /= len(val_loader)
        if scheduler is not None:
            scheduler.step(val_loss)
        val_accuracies = [c / t if t > 0 else 0.0 for c, t in zip(corrects, totals)]
        
        min_delta = 0.1  # Ajusta según el ruido de tu loss

        if best_loss - val_loss > min_delta:
            best_loss = val_loss
            best_accuracies = val_accuracies.copy()
            trigger_times = 0
            
            
            
            torch.save(
                model.state_dict(), 
                f'{trained_encoders_path}/pmodel_{best_loss}.pt')
            
            file = f'{trained_encoders_path}/pmodel_{best_loss}.txt'
            with open(file, 'w') as f:
                f.write(f'Model: {target_cols}\n')
                f.write(f'Validation Loss: {best_loss}\n')
                for i, col in enumerate(target_cols):
                    f.write(f'Best Accuracy {col}: {best_accuracies[i]}\n')
                f.write(f'Best Parameters:\n \
                        \t  {params_to_txt}\n')
                    
            
        else:
            trigger_times += 1
            if trigger_times >= patience:
                break
            
    # Devuelve la mejor loss y la mejor accuracy por cada target
    return best_loss, best_accuracies