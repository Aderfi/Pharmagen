import json
import math
import optuna
import random
import sys
import torch
import numpy as np

from .data import PGenInputDataset, PGenDataset, train_data_load
from .model import PGenModel
from .model_configs import MODEL_CONFIGS
from .train import train_model
from functools import partial
from pathlib import Path
from src.config.config import PGEN_MODEL_DIR
from torch.utils.data import DataLoader
from typing import List
import torch.nn as nn
gelu = nn.GELU()

def get_optuna_cfg(model_name):
    config = MODEL_CONFIGS[model_name]
    return [c.lower() for c in config['cols']], [t.lower() for t in config['targets']]

def get_num_drugs(data_loader):
    return len(data_loader.encoders['drug'].classes_)

def get_num_genes(data_loader):
    return len(data_loader.encoders['gene'].classes_)

def get_num_alleles(data_loader):
    return len(data_loader.encoders['allele'].classes_)

def get_num_genotypes(data_loader):
    return len(data_loader.encoders['genotype'].classes_)

def get_output_sizes(data_loader, target_cols):
    return [len(data_loader.encoders[t].classes_) for t in target_cols]

def optuna_objective(trial, model_name, target_cols=None):
    trial_seed = trial.suggest_int("trial_seed", 1, int(1e6))
    base_seed = 7
    seed = base_seed + trial_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cols, targets = get_optuna_cfg(model_name)
    if target_cols is None:
        target_cols = [t for t in targets]
    '''    
    emb_dim_drug_choices = [128, 256]
    emb_dim_gene_choices = [256, 512]
    emb_dim_allele_choices = [128, 256]
    emb_dim_geno_choices = [256, 512]
    '''
    params = {
        'emb_dim_drug': trial.suggest_categorical('emb_dim_drug', [128, 256]),
        'emb_dim_gene': trial.suggest_categorical('emb_dim_gene', [256, 512]),
        'emb_dim_allele': trial.suggest_categorical('emb_dim_allele', [128, 256]),
        'emb_dim_geno': trial.suggest_categorical('emb_dim_geno', [256, 512]),
        'hidden_dim': trial.suggest_int('hidden_dim', 768, 1024, step=128),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.35),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128]),
        'weight_decay': trial.suggest_float('weight_decay', 0.005, 0.02)
        }


    csvfiles, cols, equivalencias = train_data_load(targets)
    data_loader = PGenInputDataset()
    df = data_loader.load_data(csvfiles, cols, targets, equivalencias)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # Mezclar datos
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df, target_cols)
    val_dataset = PGenDataset(val_df, target_cols)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'])

    n_drugs = get_num_drugs(data_loader)
    n_genes = get_num_genes(data_loader)
    n_alleles = get_num_alleles(data_loader)
    n_genotypes = get_num_genotypes(data_loader)
    n_outcomes, n_variations, n_effects = get_output_sizes(data_loader, target_cols)
    
    
    model = PGenModel(
        n_drugs,
        n_genes,
        n_alleles,
        n_genotypes,
        params['emb_dim_drug'],
        params['emb_dim_gene'],
        params['emb_dim_allele'],
        params['emb_dim_geno'],
        params['hidden_dim'],
        params['dropout_rate'],
        n_outcomes, n_variations, n_effects
    )
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=12)
    criterions = [torch.nn.CrossEntropyLoss(label_smoothing=0.1), torch.nn.CrossEntropyLoss(label_smoothing=0.1), torch.nn.CrossEntropyLoss(label_smoothing=0.1), optimizer]
    epochs = 200
    patience = 20

    best_loss, best_accuracy = train_model(
        train_loader, val_loader, model, criterions,
        epochs=epochs, patience=patience, target_cols=target_cols, scheduler=scheduler
    )

    max_loss_targets = get_output_sizes(data_loader, target_cols)
    max_loss = sum([math.log(n) for n in max_loss_targets]) / len(max_loss_targets)

    trial.set_user_attr('accuracy', best_accuracy)
    trial.set_user_attr('seed', seed)

    print(
        f" Trial {trial.number}. Emb_dim_drug: {params['emb_dim_drug']}, Emb_dim_geno: {params['emb_dim_geno']}, Hidden_dim: {params['hidden_dim']}, "
        f"Dropout: {params['dropout_rate']:.4f}, LR: {params['learning_rate']:.6f}, Batch_size: {params['batch_size']} "
        f"=> Val Loss: {best_loss} // Max_Loss: {max_loss:.4f}, Accuracy: {best_accuracy}, SEED: {seed}",
        file=sys.stderr)

    return best_loss

def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    valid_trials = [t for t in study.trials if t.value is not None and isinstance(t.value, (int, float))]
    return sorted(valid_trials, key=lambda t: float(t.value))[:n]       # type: ignore

def run_optuna_with_progress(model_name, n_trials=150, output_dir=PGEN_MODEL_DIR, target_cols=None):
    output_dir = Path(output_dir)
    results_dir = output_dir / 'optuna_outputs'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f'optuna_{model_name}.json'

    cols, targets = get_optuna_cfg(model_name)
    if target_cols is None:
        target_cols = [t for t in targets]
    else:
        target_cols = [t for t in target_cols]

    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(optuna_objective, model_name=model_name, target_cols=target_cols),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_5 = get_best_trials(study, 5)

    data_loader = PGenInputDataset()
    csvfiles, cols, equivalencias = train_data_load(targets)
    df = data_loader.load_data(csvfiles, cols, targets, equivalencias)
    n_outcomes, n_variations, n_effects = get_output_sizes(data_loader, target_cols)
    max_loss = sum([math.log(n) for n in (n_outcomes, n_variations, n_effects)]) / 3.0
    normalized_loss = study.best_trial.value / max_loss      # type: ignore

    best_accuracy = study.best_trial.user_attrs.get('accuracy', None)

    output = {
        "model": model_name,
        "best_trial": {
            "loss": study.best_trial.value,
            "normalized_loss": normalized_loss,
            "accuracy": best_accuracy,
            "params": study.best_trial.params
        },
        "top5": [
            {"loss": t.value, "params": t.params}
            for t in best_5
        ],
        "n_trials": n_trials
    }
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    file = results_dir / f'optuna_{model_name}.txt'
    with open(file, "w") as f:
        f.write(f"Mejores hiperparámetros para {model_name} (loss: {study.best_trial.value}, normalized_loss: {normalized_loss}, accuracy: {best_accuracy}):\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTop 5 pruebas:\n")
        for i, t in enumerate(best_5, 1):
            f.write(f"{i}. Loss: {t.value}, Params: {t.params}\n")

    return study.best_trial.params, study.best_trial.value, results_file, normalized_loss