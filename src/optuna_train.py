#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna Hyperparameter Optimization with Multi-Objective Support.

This module implements a comprehensive hyperparameter optimization pipeline using Optuna
with support for multi-objective optimization. Key features:

- Multi-objective optimization (loss + F1 for critical tasks)
- Per-task metric tracking (F1, precision, recall)
- Proper handling of class imbalance with weighted loss
- Clinical priority weighting for task balancing
- Comprehensive reporting with JSON and visualization

References:
    - Optuna: https://optuna.readthedocs.io/
    - Multi-objective optimization: Deb et al., 2002
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna Hyperparameter Optimization Pipeline.

Refactorizado para eficiencia de memoria:
1. Carga datos y crea Datasets una sola vez (evita Data Leakage y overhead IO).
2. Pasa referencias de los datasets a la función objetivo.
3. Soporta optimización mono y multi-objetivo.
"""

import datetime
import json
import logging
import math
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
import pandas as pd
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.config import PGEN_MODEL_DIR, PROJECT_ROOT, MODELS_DIR, MODEL_TRAIN_DATA
from .data import PGenDataProcess, PGenDataset
from .model import DeepFM_PGenModel
from .model_configs import (
    MODEL_REGISTRY,
    MULTI_LABEL_COLUMN_NAMES,
    get_model_config,
)
from .train import train_model
from .train_utils import create_optimizer, create_criterions, create_scheduler

# Configure logging
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Configuration Constants
# ============================================================================

N_TRIALS = 100
EPOCHS = 50
PATIENCE = 15
RANDOM_SEED = 711

# Optuna Settings
N_STARTUP_TRIALS = 10
N_EI_CANDIDATES = 24
N_PRUNER_STARTUP_TRIALS = 10
N_PRUNER_WARMUP_STEPS = 5
PRUNER_INTERVAL_STEPS = 1

# Paths
OPTUNA_OUTPUTS = Path(PROJECT_ROOT, "optuna_outputs")
OPTUNA_FIGS = OPTUNA_OUTPUTS / "figures"
OPTUNA_DBS = OPTUNA_OUTPUTS / "study_dbs"

OPTUNA_FIGS.mkdir(parents=True, exist_ok=True)
OPTUNA_DBS.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================


def get_optuna_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Extract hyperparameter suggestions based on model config."""
    model_conf = MODEL_REGISTRY.get(model_name)
    if not model_conf:
        raise ValueError(f"Model configuration not found: {model_name}")

    search_space = model_conf.get("params_optuna")
    if not search_space:
        raise ValueError(f"No 'params_optuna' found for '{model_name}'")

    params = {}
    for param_name, space in search_space.items():
        try:
            if isinstance(space, list):
                if space and space[0] == "int":
                    # ["int", low, high, step, log]
                    low, high = space[1], space[2]
                    step = space[3] if len(space) > 3 else 1
                    log = space[4] if len(space) > 4 else False
                    params[param_name] = trial.suggest_int(
                        param_name, low, high, step=step, log=log
                    )
                else:
                    # Categorical
                    params[param_name] = trial.suggest_categorical(param_name, space)
            elif isinstance(space, tuple):
                # Float (low, high)
                low, high = space
                is_log = (
                    "rate" in param_name
                    or "decay" in param_name
                    or "dropout" in param_name
                )
                params[param_name] = trial.suggest_float(
                    param_name, low, high, log=is_log
                )
        except Exception as e:
            logger.error(f"Error parsing param '{param_name}': {e}")

    return params


def optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    feature_cols: List[str],
    target_cols: List[str],
    train_dataset: PGenDataset,
    val_dataset: PGenDataset,
    encoder_dims: Dict[str, int],
    use_multi_objective: bool = False,
) -> Tuple[float, ...]:
    """
    Función objetivo optimizada.
    Recibe Datasets ya instanciados (referencias a memoria) para evitar IO repetitivo.
    """
    # 1. Sugerir Hiperparámetros
    params = get_optuna_params(trial, model_name)
    trial.set_user_attr("full_params", params)

    # 2. DataLoaders (Ligeros)
    # Como los datasets residen en RAM (numpy), usamos workers=4 y pin_memory=True para velocidad
    batch_size = params.get("batch_size", 64)
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # 3. Configurar Dimensiones
    # Filtramos solo las dimensiones necesarias para este modelo específico
    n_features = {k: v for k, v in encoder_dims.items() if k in feature_cols}
    target_dims = {k: v for k, v in encoder_dims.items() if k in target_cols}

    # 4. Instanciar Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepFM_PGenModel(
        n_features=n_features,
        target_dims=target_dims,
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        n_layers=params["n_layers"],
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.0),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 64),
        embedding_dropout=params.get("embedding_dropout", 0.0),
    ).to(device)

    # 5. Configurar Entrenamiento
    criterions: List[Any] = create_criterions(target_cols, params, device=device)
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)
    criterions.append(optimizer)

    # 6. Ejecutar Entrenamiento
    # Para Optuna, generalmente queremos return_per_task_losses=False
    # a menos que queramos optimizar multi-objetivo específico.
    try:
        best_loss, best_accs = train_model(
            train_loader,
            val_loader,
            model,
            criterions,
            epochs=EPOCHS,
            patience=PATIENCE,
            model_name=model_name,
            feature_cols=feature_cols,
            target_cols=target_cols,
            device=device,
            scheduler=scheduler,
            trial=trial,
            multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
            return_per_task_losses=False,
            progress_bar=False,  # Desactivar barra interna para no saturar log de Optuna
        )
    except optuna.TrialPruned:
        raise
    except Exception as e:
        logger.error(f"Training failed for trial {trial.number}: {e}")
        raise e

    # Guardar métricas adicionales en el trial para reporte
    avg_acc = sum(best_accs) / len(best_accs) if best_accs else 0.0
    trial.set_user_attr("avg_accuracy", avg_acc)
    trial.set_user_attr("accuracies", best_accs)

    # Retorno
    if use_multi_objective:
        # Ejemplo: Minimizar Loss y Maximizar Accuracy (retornamos -Acc para minimizar)
        return best_loss, -avg_acc

    return (best_loss,)


def run_optuna_with_progress(
    model_name: str,
    csv_path: Path,  # Ruta explícita al archivo de datos
    n_trials: int = N_TRIALS,
    use_multi_objective: bool = False,
) -> Dict[str, Any]:

    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger.info(f"Starting Optuna for {model_name} at {datetime_str}")

    # 1. PREPARACIÓN DE DATOS (Se ejecuta UNA vez)
    # --------------------------------------------
    config = get_model_config(model_name)
    feature_cols = [c.lower() for c in config["features"]]
    target_cols = [t.lower() for t in config["targets"]]
    # Columnas necesarias para cargar (features + targets + cols extra de config)
    cols_to_load = list(set(config["cols"] + config["targets"] + config["features"]))

    logger.info("Loading and preprocessing data (One-time setup)...")

    data_proc = PGenDataProcess()
    df = data_proc.load_data(
        csv_path=csv_path,
        all_cols=cols_to_load,  # Use cols_to_load directly
        input_cols=feature_cols,
        target_cols=target_cols,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=config["stratify"],
    )

    # Split
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["stratify_col"], random_state=RANDOM_SEED
    )

    # Fit & Transform
    data_proc.fit(train_df)
    train_enc = data_proc.transform(train_df)
    val_enc = data_proc.transform(val_df)

    # Crear Datasets en Memoria (Numpy Arrays)
    train_ds = PGenDataset(
        train_enc, feature_cols, target_cols, MULTI_LABEL_COLUMN_NAMES
    )
    val_ds = PGenDataset(val_enc, feature_cols, target_cols, MULTI_LABEL_COLUMN_NAMES)

    # Extraer dimensiones globales de los encoders
    encoder_dims = {}
    for col, enc in data_proc.encoders.items():
        encoder_dims[col] = len(enc.classes_)

    logger.info("Data setup complete. Starting optimization loop...")

    # 2. CONFIGURAR ESTUDIO
    # ---------------------
    study_name = f"OPT_{model_name}_{datetime_str}"
    storage_url = f"sqlite:///{OPTUNA_DBS}/{study_name}.db"

    sampler = TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        n_ei_candidates=N_EI_CANDIDATES,
        multivariate=True,
        seed=RANDOM_SEED,
    )

    if use_multi_objective:
        directions = ["minimize", "minimize"]  # Ej: Loss, -Accuracy
        pruner = None  # Pruning no compatible con multi-obj estándar
    else:
        directions = ["minimize"]
        pruner = MedianPruner(
            n_startup_trials=N_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=N_PRUNER_WARMUP_STEPS,
            interval_steps=PRUNER_INTERVAL_STEPS,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        directions=directions,
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    # 3. EJECUTAR OPTIMIZACIÓN
    # ------------------------
    # Usamos partial para inyectar los datos estáticos
    objective_func = partial(
        optuna_objective,
        model_name=model_name,
        feature_cols=feature_cols,
        target_cols=target_cols,
        train_dataset=train_ds,
        val_dataset=val_ds,
        encoder_dims=encoder_dims,
        use_multi_objective=use_multi_objective,
    )

    # Barra de progreso externa
    with tqdm(total=n_trials, desc=f"Optuna {model_name}", colour="blue") as pbar:

        def callback(study, trial):
            pbar.update(1)
            best_val = study.best_trials[0].values[0] if study.best_trials else 0
            pbar.set_postfix(best_loss=f"{best_val:.4f}")

        study.optimize(
            objective_func, n_trials=n_trials, callbacks=[callback], gc_after_trial=True
        )

    # 4. REPORTES Y GUARDADO
    # ----------------------
    logger.info("Optimization finished. Saving results...")

    plots_saving(study, model_name, datetime_str, OPTUNA_FIGS, use_multi_objective)

    best_trials = get_best_trials(study, n=5)
    _save_optuna_report(
        study,
        model_name,
        f"report_{study_name}",
        OPTUNA_OUTPUTS,
        best_trials,
        use_multi_objective,
    )

    print(f"\nEstudio guardado en: {storage_url}")
    print(f"Mejores parámetros: {study.best_trials[0].params}")

    return study.best_trials[0].params


# ============================================================================
# Reporting Utils
# ============================================================================


def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    """Retorna los N mejores trials (o frente de Pareto)."""
    valid_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not valid_trials:
        return []

    # Si es multi-objetivo, best_trials retorna el Frente de Pareto
    # Si es single, retorna [best_trial]
    best_candidates = study.best_trials

    if len(best_candidates) >= n:
        return best_candidates[:n]

    # Si queremos más de lo que da el frente de pareto o el mejor único,
    # ordenamos por la primera función objetivo (Loss)
    return sorted(
        valid_trials, key=lambda t: t.values[0] if t.values else float("inf")
    )[:n]


def plots_saving(
    study: optuna.Study,
    model_name: str,
    datetime_str: str,
    output_dir: Path,
    is_multi: bool,
):
    """Genera y guarda gráficos usando Matplotlib backend."""
    import matplotlib.pyplot as plt
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_pareto_front,
        plot_param_importances,
        plot_slice,
    )

    plt.style.use("ggplot")
    base_filename = f"{model_name}_{datetime_str}"

    # 1. History
    try:
        plt.figure(figsize=(10, 6))
        plot_optimization_history(study)
        plt.title(f"Optimization History - {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"history_{base_filename}.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Plot Error (History): {e}")

    # 2. Pareto (Multi only)
    if is_multi:
        try:
            plt.figure(figsize=(10, 6))
            plot_pareto_front(study)
            plt.title(f"Pareto Front - {model_name}")
            plt.tight_layout()
            plt.savefig(output_dir / f"pareto_{base_filename}.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Plot Error (Pareto): {e}")

    # 3. Importance & Slice (Single only usually, or first objective)
    if not is_multi and len(study.trials) > 10:
        try:
            plt.figure(figsize=(10, 8))
            plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(output_dir / f"importance_{base_filename}.png", dpi=150)
            plt.close()

            plt.figure(figsize=(10, 8))
            plot_slice(study)
            plt.tight_layout()
            plt.savefig(output_dir / f"slice_{base_filename}.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"Plot Error (Imp/Slice): {e}")


def _save_optuna_report(study, model_name, filename, output_dir, best_trials, is_multi):
    """Guarda JSON y TXT con resumen."""
    json_path = output_dir / f"{filename}.json"
    txt_path = output_dir / f"{filename}.txt"

    # Estructura JSON
    report = {
        "model": model_name,
        "trials_total": len(study.trials),
        "trials_completed": len(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        ),
        "is_multi_objective": is_multi,
        "best_trials": [],
    }

    for t in best_trials:
        report["best_trials"].append(
            {
                "number": t.number,
                "values": t.values,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
        )

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Estructura TXT
    with open(txt_path, "w") as f:
        f.write(f"OPTUNA REPORT FOR {model_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Solutions Found: {len(best_trials)}\n\n")

        for i, t in enumerate(best_trials):
            f.write(f"Rank {i+1} (Trial {t.number}):\n")
            f.write(f"  Values: {t.values}\n")
            f.write(f"  Params: {t.params}\n")
            if "avg_accuracy" in t.user_attrs:
                f.write(f"  Avg Acc: {t.user_attrs['avg_accuracy']:.4f}\n")
            f.write("-" * 30 + "\n")

    logger.info(f"Reports saved to {output_dir}")


if __name__ == "__main__":
    # Ejemplo de uso
    # run_optuna_with_progress("Phenotype_Effect_Outcome", Path("data/my_data.tsv"))
    pass
