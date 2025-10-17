import torch

def predict_single_input(drug, gene, allele, genotype, model=None, encoders=None, target_cols=None):
    """
    drug, gene, allele, genotype: strings (raw user input, must be present in encoders)
    model: modelo PGenModel entrenado
    encoders: diccionario con encoders por columna
    target_cols: lista de nombres ['outcome', 'variation', 'effect']
    """
    if encoders is None or target_cols is None:
        raise ValueError("Se requieren encoders y target_cols (['outcome','variation','effect'])")
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    drug_idx = encoders['drug'].transform([drug])[0]
    gene_idx = encoders['gene'].transform([gene])[0]
    allele_idx = encoders['allele'].transform([allele])[0]
    geno_idx = encoders['genotype'].transform([genotype])[0]
    drug_tensor = torch.tensor([drug_idx], dtype=torch.long, device=device)
    gene_tensor = torch.tensor([gene_idx], dtype=torch.long, device=device)
    allele_tensor = torch.tensor([allele_idx], dtype=torch.long, device=device)
    geno_tensor = torch.tensor([geno_idx], dtype=torch.long, device=device)
    with torch.no_grad():
        out_logits, var_logits, eff_logits = model(drug_tensor, gene_tensor, allele_tensor, geno_tensor)
        out_idx = torch.argmax(out_logits, dim=1).item()
        var_idx = torch.argmax(var_logits, dim=1).item()
        eff_idx = torch.argmax(eff_logits, dim=1).item()
    outcome = encoders[target_cols[0]].inverse_transform([out_idx])[0]
    variation = encoders[target_cols[1]].inverse_transform([var_idx])[0]
    effect = encoders[target_cols[2]].inverse_transform([eff_idx])[0]
    result = {target_cols[0]: outcome, target_cols[1]: variation, target_cols[2]: effect}
    return result

def predict_from_file(file_path, model=None, encoders=None, target_cols=None):
    import pandas as pd
    df = pd.read_csv(file_path, sep=';', dtype=str)
    device = next(model.parameters()).device if model is not None else torch.device("cpu")
    drug_idx = encoders['drug'].transform(df["Drug"].astype(str))
    gene_idx = encoders['gene'].transform(df["Gene"].astype(str))
    allele_idx = encoders['allele'].transform(df["Allele"].astype(str))
    geno_idx = encoders['genotype'].transform(df["Genotype"].astype(str))
    drug_tensor = torch.tensor(drug_idx, dtype=torch.long, device=device)
    gene_tensor = torch.tensor(gene_idx, dtype=torch.long, device=device)
    allele_tensor = torch.tensor(allele_idx, dtype=torch.long, device=device)
    geno_tensor = torch.tensor(geno_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        out_logits, var_logits, eff_logits = model(drug_tensor, gene_tensor, allele_tensor, geno_tensor)
        out_idx = torch.argmax(out_logits, dim=1).cpu().numpy()
        var_idx = torch.argmax(var_logits, dim=1).cpu().numpy()
        eff_idx = torch.argmax(eff_logits, dim=1).cpu().numpy()
    outcomes = encoders[target_cols[0]].inverse_transform(out_idx)
    variations = encoders[target_cols[1]].inverse_transform(var_idx)
    effects = encoders[target_cols[2]].inverse_transform(eff_idx)
    results = []
    for o, v, e in zip(outcomes, variations, effects):
        results.append({target_cols[0]: o, target_cols[1]: v, target_cols[2]: e})
    return results