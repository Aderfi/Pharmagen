import torch

def train_model(
    train_loader, 
    val_loader, 
    model, 
    optimizer, 
    criterion, 
    epochs, 
    patience, 
    device=None, 
    targets=None
):
    """
    Entrena el modelo usando los targets que se le pasan como argumento.
    - targets: lista de nombres de target (en minúsculas), ej. ['outcome', 'variation']
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if targets is None:
        # Por defecto, usa todos los posibles
        targets = ['outcome', 'variation', 'effect', 'entity']
        
    best_loss = float('inf')
    trigger_times = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            drug_geno = batch['drug_geno'].to(device)
            # Solo usa los targets presentes en el batch y requeridos por el modelo
            batch_targets = {k: batch[k].to(device) for k in targets if k in batch}
            optimizer.zero_grad()
            outputs = model(drug_geno)

            loss = sum(criterion(outputs[k], batch_targets[k]) for k in outputs)
            loss.backward()     # type: ignore
            optimizer.step()
            total_loss += loss.item()     # type: ignore
        avg_loss = total_loss / len(train_loader)

        #======================
        #===== Validación =====
        #======================
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                drug_geno = batch['drug_geno'].to(device)
                batch_targets = {k: batch[k].to(device) for k in targets if k in batch}
                outputs = model(drug_geno)

                loss = sum(criterion(outputs[k], batch_targets[k]) for k in outputs)
                val_loss += loss.item()      # type: ignore
        val_loss /= len(val_loader)
        #print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                #print("Early stopping activado.")
                break
    return best_loss
