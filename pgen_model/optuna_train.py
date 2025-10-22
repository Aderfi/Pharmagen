import json
import math
import optuna
import random
import sys
import torch
import numpy as np
import warnings # Añadido

from .data import PGenInputDataset, PGenDataset, train_data_load
from .model import DeepFM_PGenModel # Asumiendo que has renombrado tu clase
from .model_configs import MODEL_CONFIGS, MULTI_LABEL_COLUMN_NAMES
from .train import train_model
from functools import partial
from pathlib import Path
from src.config.config import PGEN_MODEL_DIR
from torch.utils.data import DataLoader
from typing import List
import torch.nn as nn
gelu = nn.GELU()

N_TRIALS = 100
EPOCH = 120
PATIENCE = 25

def get_optuna_cfg(model_name):
    config = MODEL_CONFIGS[model_name]
    # La función ya devuelve minúsculas, no es necesario .lower() extra
    return [t.lower() for t in config['targets']]

def get_num_drugs(data_loader):
    return len(data_loader.encoders['drug'].classes_)

def get_num_genes(data_loader):
    return len(data_loader.encoders['gene'].classes_)

def get_num_alleles(data_loader):
    return len(data_loader.encoders['allele'].classes_)

def get_num_genotypes(data_loader):
    return len(data_loader.encoders['genotype'].classes_)

def get_output_sizes(data_loader, target_cols):
    # Esto sigue funcionando, ya que MultiLabelBinarizer.classes_
    # devuelve el número de clases (el ancho del vector binario).
    return [len(data_loader.encoders[t].classes_) for t in target_cols]

def optuna_objective(trial, model_name, target_cols=None):
    seed = 711
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    targets_from_config = get_optuna_cfg(model_name)
    if target_cols is None:
        target_cols = targets_from_config
    
    # --- 1. Definir Hiperparámetros ---
    params = {
    'embedding_dim': trial.suggest_categorical('embedding_dim', [256]),
    'batch_size': trial.suggest_categorical('batch_size', [128]),
    'hidden_dim': trial.suggest_categorical('hidden_dim', [768, 1024, 1280, 1536]),
    'dropout_rate': trial.suggest_float('dropout_rate', 0.15, 0.35, step=0.05),
    'learning_rate': trial.suggest_float('learning_rate', 2e-4, 9e-4, log=True),
    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-4, log=True)
}
    params_to_txt = params # Guardar para reporte

    # --- 2. Cargar Datos ---
    csvfiles, cols_to_read, equivalencias = train_data_load(targets_from_config)
    data_loader = PGenInputDataset()
    
    # <-- CAMBIO: Pasa la lista de columnas multi-etiqueta a load_data
    df = data_loader.load_data(
        csvfiles, 
        cols_to_read, 
        targets_from_config, 
        equivalencias,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES)
    )
    
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)
    
    # <-- CAMBIO: Pasa el conjunto de columnas multi-etiqueta a PGenDataset
    train_dataset = PGenDataset(train_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES)
    val_dataset = PGenDataset(val_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES)
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=True) # shuffle=True también en val es inusual, pero lo dejo

    # --- 3. Definir Modelo (Dinámicamente) ---
    n_drugs = get_num_drugs(data_loader)
    n_genes = get_num_genes(data_loader)
    n_alleles = get_num_alleles(data_loader)
    n_genotypes = get_num_genotypes(data_loader)
    
    n_targets_list = get_output_sizes(data_loader, target_cols)
    num_targets = len(target_cols)
    
    target_dims = {}
    for i, col_name in enumerate(target_cols):
        target_dims[col_name] = n_targets_list[i]
    
    model = DeepFM_PGenModel(
        n_drugs, n_genes, n_alleles, n_genotypes,
        params['embedding_dim'],
        params['hidden_dim'],
        params['dropout_rate'],
        target_dims=target_dims
    )
    
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # --- 4. Definir Criterio (Dinámicamente) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
    
    # <-- CAMBIO: Selecciona la función de pérdida correcta para cada target
    criterions = []
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=0.1))
    
    criterions.append(optimizer) # Añadir el optimizador al final
    
    epochs = EPOCH
    patience = PATIENCE

    # --- 5. Entrenar ---
    # <-- CAMBIO: Pasa el conjunto multi_label_cols a train_model
    best_loss, best_accuracies_list = train_model(
        train_loader, val_loader, model, criterions,
        epochs=epochs, patience=patience, target_cols=target_cols, 
        scheduler=None, params_to_txt=params_to_txt,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    # --- 6. Calcular Métricas y Guardar en Trial ---
    max_loss = sum([math.log(n) for n in n_targets_list]) / len(n_targets_list)
    normalized_loss = best_loss / max_loss
    avg_accuracy = sum(best_accuracies_list) / len(best_accuracies_list)

    trial.set_user_attr('avg_accuracy', avg_accuracy)
    trial.set_user_attr('normalized_loss', normalized_loss)
    trial.set_user_attr('all_accuracies', best_accuracies_list)
    trial.set_user_attr('seed', seed)

    print(
        f" Trial {trial.number}. BS: {params['batch_size']}, LR: {params['learning_rate']:.6f} "
        f"=> Val Loss: {best_loss:.4f} // Norm Loss: {normalized_loss:.4f}, Avg Acc: {avg_accuracy:.4f}, SEED: {seed}",
        file=sys.stderr)

    return best_loss

def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    """Obtiene los 'n' mejores trials válidos de un estudio."""
    valid_trials = [t for t in study.trials if t.value is not None and isinstance(t.value, (int, float))]
    return sorted(valid_trials, key=lambda t: float(t.values[0] if isinstance(t.values, list) else t.value))[:n] # type: ignore


def _save_optuna_report(
    study: optuna.Study,
    model_name: str,
    results_file_json: Path,
    results_file_txt: Path,
    top_5_trials: List[optuna.trial.FrozenTrial]
):
    """Función auxiliar para guardar los reportes JSON y TXT."""
    
    best_trial = study.best_trial
    
    # Extraer métricas guardadas del mejor trial
    normalized_loss = best_trial.user_attrs.get('normalized_loss', 0.0)
    avg_accuracy = best_trial.user_attrs.get('avg_accuracy', 0.0)
    
    # --- Guardar JSON ---
    output = {
        "model": model_name,
        "best_trial": {
            "loss": best_trial.value,
            "normalized_loss": normalized_loss,
            "avg_accuracy": avg_accuracy,
            "params": best_trial.params,
            "all_accuracies": best_trial.user_attrs.get('all_accuracies', [])
        },
        "top5": [
            {"loss": t.value, "params": t.params}
            for t in top_5_trials
        ],
        "n_trials": len(study.trials)
    }
    with open(results_file_json, "w") as f:
        json.dump(output, f, indent=2)

    # --- Guardar TXT ---
    with open(results_file_txt, "w") as f:
        f.write(f"Mejores hiperparámetros para {model_name} (loss: {best_trial.value}, normalized_loss: {normalized_loss:.4f}, avg_accuracy: {avg_accuracy:.4f}):\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTop 5 pruebas:\n")
        for i, t in enumerate(top_5_trials, 1):
            f.write(f"{i}. Loss: {t.value}, Params: {t.params}\n")

    print(f"\nReportes de Optuna guardados en: {results_file_json.parent}")


def run_optuna_with_progress(model_name, n_trials=N_TRIALS, output_dir=PGEN_MODEL_DIR, target_cols=None):
    """
    Ejecuta un estudio de Optuna y guarda los reportes.
    
    Versión limpia:
    1. No recarga los datos; confía en 'optuna_objective' para los cálculos.
    2. Delega la escritura de archivos a '_save_optuna_report'.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / 'optuna_outputs'
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file_json = results_dir / f'optuna_{model_name}.json'
    results_file_txt = results_dir / f'optuna_{model_name}.txt'

    # --- 1. Definir Targets ---
    # Asumiendo que get_optuna_cfg solo devuelve targets
    targets_from_config = get_optuna_cfg(model_name)
    if target_cols is None:
        target_cols = targets_from_config
    # (Ahora 'target_cols' es la lista de nombres de targets a usar)

    # --- 2. Ejecutar Estudio ---
    study = optuna.create_study(direction="minimize")
    study.optimize(
        partial(optuna_objective, model_name=model_name, target_cols=target_cols),
        n_trials=n_trials,
        show_progress_bar=True
    )

    # --- 3. Obtener y Guardar Resultados ---
    best_5 = get_best_trials(study, 5)

    _save_optuna_report(
        study,
        model_name,
        results_file_json,
        results_file_txt,
        best_5
    )
    
    # Extraer métricas del mejor trial para el return
    best_trial = study.best_trial
    best_loss = best_trial.value
    best_params = best_trial.params
    normalized_loss = best_trial.user_attrs.get('normalized_loss', 0.0)

    return best_params, best_loss, results_file_json, normalized_loss