import json
import math
import optuna
import random
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

def get_optuna_cfg(model_name):
    config = MODEL_CONFIGS[model_name]
    return config['cols'], config['targets']

def get_num_drug_genos(data_loader):
    return len(data_loader.encoders['Drug_Geno'].classes_)

def optuna_objective(trial, model_name):
    cols, targets = get_optuna_cfg(model_name)
    emb_dim_choices = [64, 128, 256, 512]
    params = {
        'emb_dim': trial.suggest_categorical('emb_dim', emb_dim_choices),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])

    seed = 27
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data and fit encoders
    csvfiles, _, _, _ = train_data_load(targets)
    data_loader = PGenInputDataset()
    df = data_loader.load_data(csvfiles, cols)
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
 
    n_drug_genos = get_num_drug_genos(data_loader)
    output_dims = {col.lower(): len(data_loader.encoders[col].classes_) for col in targets}

    model = PGenModel(
        n_drug_genos, params['emb_dim'], params['hidden_dim'], params['dropout_rate'], output_dims
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 120
    patience = 10 

    best_loss = train_model(
        train_loader, val_loader, model, optimizer, criterion,
        epochs=epochs, patience=patience, targets=[t.lower() for t in targets]
    )
    return best_loss

def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    valid_trials = [t for t in study.trials if t.value is not None and isinstance(t.value, (int, float))]
    return sorted(valid_trials, key=lambda t: float(t.value))[:n]       # type: ignore

def run_optuna_with_progress(model_name, n_trials=100, output_dir=PGEN_MODEL_DIR):
    # Preparar carpeta de salida
    output_dir = Path(output_dir)
    results_dir = output_dir / 'optuna_outputs'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / f'optuna_{model_name}.json'

    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(optuna_objective, model_name=model_name),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Guardar resultados finales
    best_5 = get_best_trials(study, 5)
    
    cols, targets = MODEL_CONFIGS[model_name]['cols'], MODEL_CONFIGS[model_name]['targets']
    main_target = targets[0]
    data_loader = PGenInputDataset()
    csvfiles, _, _, _ = train_data_load(targets)
    df = data_loader.load_data(csvfiles, cols)
    num_classes = len(data_loader.encoders[main_target].classes_)
    max_loss = math.log(num_classes) if num_classes > 1 else 1.0
    normalized_loss = study.best_trial.value / max_loss      # type: ignore
    
    output = {
        "model": model_name,
        "best_trial": {
            "loss": study.best_trial.value,
            "normalized_loss": normalized_loss,
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
    return study.best_trial.params, study.best_trial.value, results_file, normalized_loss
