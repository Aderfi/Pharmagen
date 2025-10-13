import torch

def train_model(train_loader, val_loader, model, optimizer, criterion, epochs, patience, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    #=========================
    #===== Entrenamiento =====
    #=========================
        
    best_loss = float('inf')
    trigger_times = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            drug = batch['drug'].to(device)
            genotype = batch['genotype'].to(device)
            targets = {k: batch[k].to(device) for k in ['outcome', 'variation', 'effect', 'entity']}
            optimizer.zero_grad()
            outputs = model(drug, genotype)
            loss = sum(criterion(outputs[k], targets[k]) for k in outputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
        
    #======================
    #===== Validación =====
    #======================
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug'].to(device)
                genotype = batch['genotype'].to(device)
                targets = {k: batch[k].to(device) for k in ['effect', 'entity']}
                outputs = model(drug, genotype)
                loss = sum(criterion(outputs[k], targets[k]) for k in ['effect', 'entity'])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping activado.")
                break
    return best_loss