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

import datetime
import json
import logging
import math
import random
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import multiprocessing

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_pareto_front
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.config import PGEN_MODEL_DIR, PROJECT_ROOT, MODELS_DIR
from .data import PGenDataProcess, PGenDataset, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import (
    CLINICAL_PRIORITIES,
    MODEL_REGISTRY,
    MULTI_LABEL_COLUMN_NAMES,
    get_model_config,
)
from .train import train_model
from .train_utils import (
    get_input_dims, 
    get_output_sizes, 
    calculate_task_metrics, 
    create_optimizer, 
    create_criterions,
    create_scheduler,
    )


# Configure logging
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Configuration Constants
# ============================================================================

# Optuna optimization settings
N_TRIALS = 300
EPOCH = 150
PATIENCE = 20

# Random seed for reproducibility
RANDOM_SEED = 711

# Data splitting
VALIDATION_SPLIT = 0.2

# Loss function settings
LABEL_SMOOTHING = 0.12

# Sampler configuration
N_STARTUP_TRIALS = 15
N_EI_CANDIDATES = 24

# Pruner configuration
N_PRUNER_STARTUP_TRIALS = 10
N_PRUNER_WARMUP_STEPS = 5
PRUNER_INTERVAL_STEPS = 3

# Class weighting
CLASS_WEIGHT_LOG_SMOOTHING = 2

# Output directory
optuna_results = Path(PROJECT_ROOT, "optuna_outputs", "figures")
optuna_results.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================

def get_optuna_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Enhanced parameter suggestion with new search space."""
    model_conf = MODEL_REGISTRY.get(model_name)
    if not model_conf:
        raise ValueError(f"Model configuration not found: {model_name}")

    search_space = model_conf.get("params_optuna")
    if not search_space:
        logger.warning(f"No 'params_optuna' found for '{model_name}'")
        return model_conf.get("params", {}).copy()

    suggested_params = {}

    for param_name, space_definition in search_space.items():
        try:
            if isinstance(space_definition, list):
                if space_definition and space_definition[0] == "int":
                    # Integer parameter: ["int", low, high, step(optional), log(optional)]
                    _, low, high = space_definition[:3]
                    step = space_definition[3] if len(space_definition) > 3 else 1
                    log = space_definition[4] if len(space_definition) > 4 else False
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, int(low), int(high), step=int(step), log=bool(log)
                    )
                else:
                    # Categorical parameter
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name, space_definition
                    )
            elif isinstance(space_definition, tuple):
                # Float parameter: (low, high)
                low, high = space_definition
                # Determine if log scale should be used
                is_log_scale = param_name in [
                    "learning_rate", "weight_decay", "attention_dropout",
                    "dropout_rate", "fm_dropout", "embedding_dropout"
                ]
                suggested_params[param_name] = trial.suggest_float(
                    param_name, low, high, log=is_log_scale
                )
            else:
                logger.warning(f"Unknown search space type for '{param_name}': {type(space_definition)}")
        
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error parsing search space for '{param_name}': {e}")

    return suggested_params


def optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    pbar: tqdm,
    target_cols: List[str],
    fitted_data_loader: PGenDataProcess,
    train_processed_df: pd.DataFrame,
    val_processed_df: pd.DataFrame,
    csvfiles: Path,
    class_weights_task3: Optional[torch.Tensor] = None,
    use_multi_objective: bool = True,
) -> Tuple[float, ...]:
    """
    Optuna objective function for hyperparameter optimization.
    
    Trains a model with suggested hyperparameters and evaluates it on validation data.
    Supports both single and multi-objective optimization.
    
    Args:
        trial: Optuna trial object
        model_name: Name of the model to train
        pbar: Progress bar for updates
        target_cols: List of target column names
        fitted_data_loader: Fitted PGenDataProcess with encoders
        train_processed_df: Processed training DataFrame
        val_processed_df: Processed validation DataFrame
        csvfiles: Path to CSV files (for logging)
        class_weights_task3: Optional class weights for imbalanced tasks
        use_multi_objective: If True, optimize (loss, -F1). If False, only loss.
    
    Returns:
        Tuple of objective values:
        - If multi-objective: (loss, -critical_f1)
        - If single-objective: (loss,)
    """
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get suggested hyperparameters
    full_base_config = get_model_config(model_name)
    base_params = full_base_config.get("params", {})
    
    # Get the parameters suggested by Optuna for this specific trial
    optuna_suggested_params = get_optuna_params(trial, model_name)
    
    # Merge them: Optuna's suggestions will overwrite the base parameters
    params = {**base_params, **optuna_suggested_params}
    
    # Save the complete and correct set of parameters to the trial for reporting
    trial.set_user_attr("full_params", params)
    

    # 2. Create datasets and dataloaders
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    # Optimize DataLoader configuration
    batch_size = params.get("batch_size", 64)
    num_workers = min(multiprocessing.cpu_count(), 8)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    # 3. Create model
    input_dims = get_input_dims(fitted_data_loader)
    n_targets_list = get_output_sizes(fitted_data_loader, target_cols)
    target_dims = {col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)}

    separate_embedding_dims = None
    if any(key.endswith("_embedding_dim") for key in params):
        separate_embedding_dims = {
            "drug": params.get("drug_embedding_dim", params["embedding_dim"]),
            "gene": params.get("gene_embedding_dim", params["embedding_dim"]),
            "allele": params.get("allele_embedding_dim", params["embedding_dim"]),
            "genalle": params.get("genalle_embedding_dim", params["embedding_dim"]),
        }

    if trial.number == 0:
        logger.info(f"Loading data from: {Path(csvfiles)}")
        logger.info(f"Tasks: {', '.join(target_cols)}")
        logger.info(f"Clinical priorities: {CLINICAL_PRIORITIES}")
        
    model = DeepFM_PGenModel(
        n_drugs=input_dims["drug"],
        n_genalles=input_dims["genalle"],
        n_genes=input_dims["gene"],
        n_alleles=input_dims["allele"],
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        target_dims=target_dims,
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.1),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 256),
        embedding_dropout=params.get("embedding_dropout", 0.1),
        separate_emb_dims_dict=separate_embedding_dims if separate_embedding_dims != None else None,    
    ).to(device)
    
    # 4. Define optimizer and loss functions
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)
    
    criterions = create_criterions(target_cols, params, class_weights_task3, device)
    criterions.append(optimizer)
    
    epochs = params.get("epochs", EPOCH)
    patience = params.get("patience", PATIENCE)
    
    if "gradient_clip_norm" in params:
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip_norm"])
        
    # 5. Train model
    best_loss, best_accuracies_list = train_model( # type: ignore
        train_loader, 
        val_loader, 
        model, 
        criterions,
        epochs=epochs, 
        patience=patience, 
        target_cols=target_cols,
        scheduler=scheduler, 
        params_to_txt=params,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        progress_bar=False, 
        model_name=model_name,
        trial=trial,
    )
    # 6. Calculate detailed metrics
    task_metrics = calculate_task_metrics(
        model, val_loader, target_cols, MULTI_LABEL_COLUMN_NAMES, device
    )

    # 7. Calculate normalized loss
    max_loss_list = []
    for i, col in enumerate(target_cols):
        if col in MULTI_LABEL_COLUMN_NAMES:
            max_loss_list.append(math.log(2))
        else:
            n_classes = n_targets_list[i]
            max_loss_list.append(math.log(n_classes) if n_classes > 1 else 0.0)

    max_loss = sum(max_loss_list) if max_loss_list else 1.0
    normalized_loss = best_loss / max_loss if max_loss > 0 else best_loss

    avg_accuracy = (
        sum(best_accuracies_list) / len(best_accuracies_list)
        if best_accuracies_list
        else 0.0
    )

    # 8. Store metrics in trial
    trial.set_user_attr("avg_accuracy", avg_accuracy)
    trial.set_user_attr("normalized_loss", normalized_loss)
    trial.set_user_attr("all_accuracies", best_accuracies_list)
    trial.set_user_attr("seed", RANDOM_SEED)

    # Store per-task metrics
    for col, metrics in task_metrics.items():
        for metric_name, value in metrics.items():
            trial.set_user_attr(f"{col}_{metric_name}", value)

    # 9. Identify critical task metric
    critical_task = "effect_type"
    critical_f1 = task_metrics.get(critical_task, {}).get("f1_weighted", 0.0)

    trial.set_user_attr("critical_task_f1", critical_f1)
    trial.set_user_attr("critical_task_name", critical_task)

    # 10. Update progress bar
    info_dict = {
        "Trial": trial.number,
        "Loss": f"{best_loss:.4f}",
        "NormLoss": f"{normalized_loss:.4f}",
        f"{critical_task}_F1": f"{critical_f1:.4f}",
        "AvgAcc": f"{avg_accuracy:.4f}",
    }

    pbar.set_postfix(info_dict, refresh=True)
    pbar.update(1)

    # 11. Return objectives
    if use_multi_objective:
        # Multi-objective: minimize loss AND maximize F1 (negated for minimization)
        return best_loss, -critical_f1
    else:
        # Single-objective: only loss
        return (best_loss,)


def get_best_trials(
    study: optuna.Study, n: int = 5
) -> List[optuna.trial.FrozenTrial]:
    """
    Get the top N valid trials from a study.
    
    Handles both single and multi-objective studies. For multi-objective,
    returns trials from the Pareto front.
    
    Args:
        study: Optuna study object
        n: Number of trials to return
    
    Returns:
        List of top N trials, sorted by objective values
    """
    valid_trials = [
        t for t in study.trials
        if t.values is not None and 
           all(isinstance(v, (int, float)) and not math.isnan(v) for v in t.values) and
           t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not valid_trials:
        logger.warning("No valid completed trials found in study")
        return []

    # Para multi-objetivo, usar best_trials (Pareto front)
    if len(valid_trials[0].values) > 1:
        try:
            # Obtener trials del frente de Pareto
            pareto_trials = study.best_trials
            # Limitar a N trials
            return pareto_trials[:n] if pareto_trials else valid_trials[:n]
        except Exception as e:
            logger.warning(f"Error getting Pareto trials: {e}")
            # Fallback: ordenar por primera métrica
            return sorted(valid_trials, key=lambda t: t.values[0])[:n]
    else:
        # Para single-objetivo, ordenar por valor
        return sorted(valid_trials, key=lambda t: t.values[0])[:n]


def _save_optuna_report(
    study: optuna.Study,
    model_name: str,
    results_file_json: Path,
    results_file_txt: Path,
    top_5_trials: List[optuna.trial.FrozenTrial],
    is_multi_objective: bool = False,
) -> None:
    """
    Save Optuna optimization results to JSON and TXT files.
    
    Args:
        study: Optuna study object
        model_name: Name of the model
        results_file_json: Path to save JSON report
        results_file_txt: Path to save TXT report
        top_5_trials: List of top trials to include
        is_multi_objective: Whether study was multi-objective
    
    Raises:
        IOError: If files cannot be written
    """
    if not top_5_trials:
        logger.warning("No valid trials found to save in report")
        return

    # Para multi-objetivo, el "mejor" trial es subjetivo, tomar el primero del Pareto front
    best_trial = top_5_trials[0]

    # Save JSON
    output = {
        "model": model_name,
        "optimization_type": "multi_objective" if is_multi_objective else "single_objective",
        "n_trials": len(study.trials),
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial": {
            "number": best_trial.number,
            "values": list(best_trial.values) if best_trial.values else [],
            "params": best_trial.user_attrs.get("full_params", best_trial.params),
            "user_attrs": dict(best_trial.user_attrs),
        },
        "clinical_priorities": CLINICAL_PRIORITIES,
    }
    
    if is_multi_objective:
        # Para multi-objetivo, incluir todos los trials del Pareto front
        output["pareto_front"] = [
            {
                "number": t.number,
                "values": list(t.values) if t.values else [],
                "params": t.user_attrs.get("full_params", t.params),
                "critical_f1": t.user_attrs.get("critical_task_f1", 0.0),
            }
            for t in top_5_trials
        ]
    else:
        # Para single-objetivo, incluir top 5 por loss
        output["top5"] = [
            {
                "number": t.number,
                "values": list(t.values) if t.values else [],
                "params": t.params,
                "critical_f1": t.user_attrs.get("critical_task_f1", 0.0),
            }
            for t in top_5_trials
        ]

    try:
        with open(results_file_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"JSON report saved: {results_file_json}")
    except IOError as e:
        logger.error(f"Failed to save JSON report: {e}")
        raise

    # Save TXT
    try:
        with open(results_file_txt, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"OPTIMIZATION REPORT: {model_name}\n")
            f.write("=" * 70 + "\n\n")

            if is_multi_objective:
                f.write("Optimization Type: Multi-Objective\n")
                f.write("   - Objective 1: Minimize Loss\n")
                f.write(f"   - Objective 2: Maximize F1 ({best_trial.user_attrs.get('critical_task_name', 'N/A')})\n\n")
                f.write("Note: Results show Pareto-optimal solutions\n\n")
            else:
                f.write("Optimization Type: Single-Objective (Loss)\n\n")

            f.write(f"Representative Trial (#{best_trial.number})\n")
            f.write("-" * 60 + "\n\n")

            # Objective values
            if best_trial.values:
                f.write("Objective Values:\n")
                if is_multi_objective:
                    f.write(f"   Loss:         {best_trial.values[0]:.5f}\n")
                    f.write(f"   F1 (negated): {best_trial.values[1]:.5f} → F1 actual: {-best_trial.values[1]:.5f}\n")
                else:
                    f.write(f"   Loss:         {best_trial.values[0]:.5f}\n")
                f.write("\n")

            # Detailed metrics
            f.write("Detailed Metrics:\n")
            f.write(f"   Normalized Loss:    {best_trial.user_attrs.get('normalized_loss', 0.0):.5f}\n")
            f.write(f"   Average Accuracy:   {best_trial.user_attrs.get('avg_accuracy', 0.0):.4f}\n")
            f.write(f"   Critical Task F1:   {best_trial.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
            f.write("\n")

            # Hyperparameters
            f.write("Hyperparameters:\n")
            # Use full_params from user_attrs if available, otherwise fallback to trial.params
            report_params = best_trial.user_attrs.get("full_params", best_trial.params)
            for key, value in sorted(report_params.items()):
                f.write(f"   {key:25s}: {value}\n")
            f.write("\n")

            # Top results
            if is_multi_objective:
                f.write("Pareto Front (Top Solutions)\n")
                f.write("-" * 60 + "\n\n")
                for i, t in enumerate(top_5_trials, 1):
                    f.write(f"{i}. Trial #{t.number}\n")
                    if t.values:
                        f.write(f"   Loss: {t.values[0]:.5f}, F1: {-t.values[1]:.5f}\n")
                    f.write(f"   Critical F1: {t.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
                    all_trial_params = t.user_attrs.get("full_params", t.params)
                    f.write(f"   Key params: batch_size={all_trial_params.get('batch_size')}, lr={all_trial_params.get('learning_rate'):.2e}\n\n")
                                        
            else:
                f.write("Top 5 Trials by Loss\n")
                f.write("-" * 60 + "\n\n")
                for i, t in enumerate(top_5_trials, 1):
                    f.write(f"{i}. Trial #{t.number}\n")
                    if t.values:
                        f.write(f"   Loss: {t.values[0]:.5f}\n")
                    f.write(f"   Critical F1: {t.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
                    all_trial_params = t.user_attrs.get("full_params", t.params)
                    f.write(f"   Key params: batch_size={all_trial_params.get('batch_size')}, lr={all_trial_params.get('learning_rate'):.2e}\n\n")
                    
                    
        logger.info(f"TXT report saved: {results_file_txt}")
    except IOError as e:
        logger.error(f"Failed to save TXT report: {e}")
        raise


def run_optuna_with_progress(
    model_name: str,
    n_trials: int = N_TRIALS,
    output_dir: Path = PGEN_MODEL_DIR,
    target_cols: Optional[List[str]] = None,
    use_multi_objective: bool = True,
) -> Tuple[Dict[str, Any], float, Optional[Path], float]:
    """
    Run Optuna hyperparameter optimization with progress tracking.
    
    Executes a complete optimization study with support for single and
    multi-objective optimization. Generates visualizations and reports.
    
    Args:
        model_name: Name of the model to optimize
        n_trials: Number of trials to run
        output_dir: Directory to save results
        target_cols: List of target columns (if None, uses config)
        use_multi_objective: If True, optimize (loss, -F1). If False, only loss.
    
    Returns:
        Tuple of (best_params, best_loss, results_json_path, normalized_loss)
        Returns empty dict and None values if optimization fails.
    
    Raises:
        ValueError: If model_name not found or n_trials invalid
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    datetime_study = datetime.datetime.now().strftime("%d_%m_%y__%H_%M")
    output_dir = Path(output_dir)
    results_dir = output_dir / "optuna_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define targets
    config = get_model_config(model_name)
    inputs_from_config = [i.lower() for i in config["inputs"]]
    targets_from_config = [t.lower() for t in config["targets"]]
    cols_from_config = config["cols"]

    if target_cols is None:
        target_cols = targets_from_config

    # 2. Load and preprocess data
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION - PHARMAGEN PMODEL")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(
        f"Mode: {'Multi-Objective (Loss + F1)' if use_multi_objective else 'Single-Objective (Loss)'}"
    )
    print(f"Trials: {n_trials}")
    print("=" * 60 + "\n")

    logger.info("Loading and preprocessing data...")
    csvfiles, _ = train_data_import(targets_from_config)

    ###################################
    
    ####################################
    '''
    data_loader = PGenDataProcess(
        all_cols=cols_from_config,
        input_cols=inputs_from_config,
        target_cols=targets_from_config,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    
    df = data_loader.load_and_clean_data(
    csv_path=str(csvfiles),
    stratify_cols=["effect_type"]
    )
    '''
    data_loader = PGenDataProcess()
    df = data_loader.load_data(
        csv_path=str(csvfiles),
        all_cols=cols_from_config,
        input_cols=inputs_from_config,
        target_cols=targets_from_config,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=["effect_type"]
    )
    
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["stratify_col"],
    )

    data_loader.fit(train_df)
    train_processed_df = data_loader.transform(train_df)
    val_processed_df = data_loader.transform(val_df)
    logger.info("Data ready.")

    # 3. Calculate class weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable GPU optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("GPU optimizations enabled (cuDNN benchmark, TF32)")
    
    class_weights_task3 = None
    task3_name = "effect_type"
    # La variable 'targets_from_config' viene de tu archivo de configuración
    # La variable 'device' ya debería estar definida (ej: torch.device("cuda"))
    # La constante CLASS_WEIGHT_LOG_SMOOTHING también debe estar definida (ej: 1.01)

    # Comprueba si la tarea está activa en la configuración actual
    if task3_name in targets_from_config:
        logger.info(f"Calculando pesos de clase para la tarea '{task3_name}'...")
        try:
            encoder_task3 = data_loader.encoders[task3_name]
            class_counts = train_processed_df[task3_name].value_counts()
            num_classes = len(encoder_task3.classes_)
            all_counts = torch.zeros(num_classes)
            
            all_counts[class_counts.index] = torch.tensor(class_counts.values, dtype=torch.float32) # type: ignore
            weights = 1.0 / torch.log(all_counts + CLASS_WEIGHT_LOG_SMOOTHING)
            
            weights[torch.isinf(weights)] = 1.0 # Asigna un peso neutral a clases no vistas en train
            weights[torch.isnan(weights)] = 1.0 # Por si acaso

            weights = weights / weights.mean()
            class_weights_task3 = weights.to(device)

            logger.info(
                f"Pesos de clase calculados para '{task3_name}' ({num_classes} clases): "
                f"Min={weights.min():.3f}, Max={weights.max():.3f}, Mean={weights.mean():.3f}"
            )
        except KeyError:
            logger.warning(
                f"No se pudo calcular los pesos: el encoder para '{task3_name}' no fue encontrado. "
                "Asegúrate de que esté en 'target_cols' y presente en los datos de entrenamiento."
            )
        except Exception as e:
            logger.error(f"Ocurrió un error inesperado al calcular los pesos de clase: {e}")

    # 4. Configure sampler
    sampler = TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        n_ei_candidates=N_EI_CANDIDATES,
        multivariate=True,
        seed=RANDOM_SEED,
    )

    # 5. Run optimization
    pbar = tqdm(
        total=n_trials,
        desc=f"Optuna ({model_name})",
        colour="green",
        leave=False,
        unit="trial",
    )

    optuna_func = partial(
        optuna_objective,
        model_name=model_name,
        target_cols=target_cols,
        pbar=pbar,
        fitted_data_loader=data_loader,
        train_processed_df=train_processed_df,
        val_processed_df=val_processed_df,
        csvfiles=csvfiles,
        class_weights_task3=class_weights_task3,
        use_multi_objective=use_multi_objective,
    )

    
    study_name=f"optuna_{model_name}_{datetime_study}"
    
    # Create study based on mode
    if use_multi_objective:
        pruner = None
        logger.info("Pruning disabled (incompatible with multi-objective)")
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name=study_name,
            storage = f"sqlite:///{MODELS_DIR}/{study_name}.db",
            sampler=sampler,
            pruner=pruner,
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=N_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=N_PRUNER_WARMUP_STEPS,
            interval_steps=PRUNER_INTERVAL_STEPS,
        )
        logger.info("Pruning enabled (single-objective)")
        study = optuna.create_study(
            direction="minimize",
            study_name=f"optuna_{model_name}_{datetime_study}",
            sampler=sampler,
            pruner=pruner,
        )

    study.optimize(optuna_func, n_trials=n_trials, gc_after_trial=True)
    
    
    pbar.close()

    # 6. Save visualizations
    if study.trials:
        try:
            # Optimization history
            fig_history = plot_optimization_history(study)
            graph_filename = optuna_results / f"history_{model_name}_{datetime_study}.html"
            fig_history.write_html(str(graph_filename))
            logger.info(f"Optimization history saved: {graph_filename}")

            # Pareto front (multi-objective only)
            if use_multi_objective:
                # Para multi-objetivo, usamos study.best_trials en lugar de study.best_trial
                if len(study.best_trials) > 1:
                    try:
                        fig_pareto = plot_pareto_front(study)
                        pareto_filename = optuna_results / f"pareto_{model_name}_{datetime_study}.html"
                        fig_pareto.write_html(str(pareto_filename))
                        logger.info(f"Pareto front saved: {pareto_filename}")
                    except Exception as pareto_error:
                        logger.warning(f"Error creating Pareto front plot: {pareto_error}")
                else:
                    logger.info("Not enough trials for Pareto front visualization (need at least 2)")
            
            # Visualizaciones adicionales para ambos modos
            try:
                # Parameter importances (solo funciona con single-objective)
                if not use_multi_objective and len(study.trials) >= 10:
                    from optuna.visualization import plot_param_importances
                    fig_importance = plot_param_importances(study)
                    importance_filename = optuna_results / f"param_importance_{model_name}_{datetime_study}.html"
                    fig_importance.write_html(str(importance_filename))
                    logger.info(f"Parameter importance saved: {importance_filename}")
            except Exception as importance_error:
                logger.debug(f"Could not create parameter importance plot: {importance_error}")
                
            try:
                # Slice plot (funciona para ambos modos)
                from optuna.visualization import plot_slice
                fig_slice = plot_slice(study)
                slice_filename = optuna_results / f"slice_{model_name}_{datetime_study}.html"
                fig_slice.write_html(str(slice_filename))
                logger.info(f"Slice plot saved: {slice_filename}")
            except Exception as slice_error:
                logger.debug(f"Could not create slice plot: {slice_error}")

        except Exception as e:
            logger.warning(f"Error saving visualizations: {e}")

    # 7. Get and save report
    best_5 = get_best_trials(study, 5)

    if not best_5:
        logger.error("Optimization completed but no valid trials found")
        return {}, float("inf"), None, 0.0

    filename = f"optuna_{model_name}_{datetime_study}"
    results_file_json = results_dir / f"{filename}.json"
    results_file_txt = results_dir / f"{filename}.txt"

    _save_optuna_report(
        study,
        model_name,
        results_file_json,
        results_file_txt,
        best_5,
        is_multi_objective=use_multi_objective,
    )

    # Extract metrics from best trial
    best_trial = best_5[0]
    best_params = best_trial.params
    best_loss = best_trial.values[0] if best_trial.values else float("inf")
    normalized_loss = best_trial.user_attrs.get("normalized_loss", 0.0)
    best_f1 = best_trial.user_attrs.get("critical_task_f1", 0.0)

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETED")
    print("=" * 70)
    
    if use_multi_objective:
        print(f"Mode: Multi-Objective Optimization")
        print(f"Pareto Front Solutions: {len(study.best_trials)}")
        print(f"\nRepresentative Solution (Trial #{best_trial.number}):")
        print(f"   Loss:           {best_loss:.5f}")
        print(f"   F1 (Critical):  {best_f1:.5f}")
        if len(best_trial.values) > 1:
            print(f"   F1 (Objective): {-best_trial.values[1]:.5f}")
    else:
        print(f"Mode: Single-Objective Optimization")
        print(f"Best Trial: #{best_trial.number}")
        print(f"   Loss:           {best_loss:.5f}")
        print(f"   F1 (Critical):  {best_f1:.5f}")

    print(f"\nCompleted Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{len(study.trials)}")
    print(f"Reports saved in: {results_dir}")
    print("=" * 70 + "\n")

    return best_params, best_loss, results_file_json, normalized_loss
