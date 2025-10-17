# coding=utf-8
import torch
import numpy as np
import random

from torch.utils.data import DataLoader
from .data import PGenInputDataset, PGenDataset, train_data_load
from .model import PGenModel
from .train import train_model
from .metrics import *
from .model_configs import MODEL_CONFIGS

def train_pipeline(PMODEL_DIR, csv_files, model_name, params, epochs=100, patience=5, batch_size=8, target_cols=None):
    seed = 27
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = MODEL_CONFIGS[model_name]
    cols = config['cols']
    targets = config['targets']
    if target_cols is None:
        target_cols = [t.lower() for t in targets]
    else:
        target_cols = [t.lower() for t in target_cols]

    csvfiles, read_cols, equivalencias = train_data_load(targets)
    data_loader = PGenInputDataset()
    df = data_loader.load_data(csv_files, cols, targets, equivalencias)
    print(f"Semilla utilizada: {seed}")
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df, target_cols)
    val_dataset = PGenDataset(val_df, target_cols)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_drugs = len(data_loader.encoders['drug'].classes_)
    n_genes = len(data_loader.encoders['gene'].classes_)
    n_alleles = len(data_loader.encoders['allele'].classes_)
    n_genotypes = len(data_loader.encoders['genotype'].classes_)
    n_outcomes = len(data_loader.encoders[target_cols[0]].classes_)
    n_variations = len(data_loader.encoders[target_cols[1]].classes_)
    n_effects = len(data_loader.encoders[target_cols[2]].classes_)

    model = PGenModel(
        n_drugs, n_genes, n_alleles, n_genotypes,
        params['emb_dim_drug'], params['emb_dim_gene'], params['emb_dim_allele'], params['emb_dim_geno'],
        params['hidden_dim'], params['dropout_rate'],
        n_outcomes, n_variations, n_effects
    )
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=0.01)
    criterions = [torch.nn.CrossEntropyLoss(label_smoothing=0.1), torch.nn.CrossEntropyLoss(label_smoothing=0.1), torch.nn.CrossEntropyLoss(label_smoothing=0.1), optimizer]
    best_loss, best_accuracy = train_model(
        train_loader, val_loader, model, criterions,
        epochs=epochs, patience=patience, target_cols=target_cols
    )
    print("Mejor loss en validación:", best_loss)
    import os
    results_dir = os.path.join(PMODEL_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(f'{results_dir}/training_report.txt', 'w') as f:
        f.write(f"Mejor loss en validación: {best_loss}\n")
        f.write(f"Hiperparámetros:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    return model