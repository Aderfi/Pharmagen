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

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_pareto_front
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.config.config import PGEN_MODEL_DIR, PROJECT_ROOT
from .data import PGenDataProcess, PGenDataset, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import (
    CLINICAL_PRIORITIES,
    MODEL_REGISTRY,
    MULTI_LABEL_COLUMN_NAMES,
    get_model_config,
)
from .train import train_model

# Configure logging
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# Configuration Constants
# ============================================================================

# Optuna optimization settings
N_TRIALS = 100
EPOCH = 100
PATIENCE = 18

# Random seed for reproducibility
RANDOM_SEED = 711

# Data splitting
VALIDATION_SPLIT = 0.2

# Loss function settings
LABEL_SMOOTHING = 0.1

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
    """
    Suggest hyperparameters for an Optuna trial based on dynamic configuration.
    
    Reads from MODEL_REGISTRY and suggests parameters according to their
    search space definitions. Supports int, float, and categorical parameters.
    
    Args:
        trial: Optuna trial object
        model_name: Name of the model configuration to use
    
    Returns:
        Dictionary mapping parameter names to suggested values
    
    Raises:
        ValueError: If model_name not found in MODEL_REGISTRY
        KeyError: If required configuration keys are missing
    """
    model_conf = MODEL_REGISTRY.get(model_name)

    if not model_conf:
        raise ValueError(
            f"Model configuration not found: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    search_space = model_conf.get("params_optuna")
    if not search_space:
        logger.warning(
            f"No 'params_optuna' found for '{model_name}'. "
            f"Using fixed 'params' from registry."
        )
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
                is_log_scale = param_name in ["learning_rate", "weight_decay"]
                suggested_params[param_name] = trial.suggest_float(
                    param_name, low, high, log=is_log_scale
                )
            else:
                logger.warning(
                    f"Unknown search space type for '{param_name}': "
                    f"{type(space_definition)}. Skipping."
                )
        except (ValueError, TypeError, IndexError) as e:
            logger.error(
                f"Error parsing search space for '{param_name}': {e}. Skipping."
            )

    return suggested_params


def get_input_dims(data_loader: PGenDataProcess) -> Dict[str, int]:
    """
    Get vocabulary sizes for all input columns.
    
    Args:
        data_loader: Fitted PGenDataProcess instance with encoders
    
    Returns:
        Dictionary mapping input column names to vocabulary sizes
    
    Raises:
        KeyError: If required encoder not found
        AttributeError: If encoder missing 'classes_' attribute
    """
    input_cols = ["drug", "genalle", "gene", "allele"]
    dims = {}
    
    for col in input_cols:
        try:
            if col not in data_loader.encoders:
                raise KeyError(f"Encoder not found for input column: {col}")
            
            encoder = data_loader.encoders[col]
            if not hasattr(encoder, "classes_"):
                raise AttributeError(f"Encoder for '{col}' missing 'classes_' attribute")
            
            dims[col] = len(encoder.classes_)
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting input dimension for '{col}': {e}")
            raise

    return dims


def get_output_sizes(
    data_loader: PGenDataProcess, target_cols: List[str]
) -> List[int]:
    """
    Get vocabulary sizes for all target columns.
    
    Args:
        data_loader: Fitted PGenDataProcess instance with encoders
        target_cols: List of target column names
    
    Returns:
        List of vocabulary sizes corresponding to target_cols
    
    Raises:
        KeyError: If target encoder not found
    """
    sizes = []
    for col in target_cols:
        try:
            if col not in data_loader.encoders:
                raise KeyError(f"Encoder not found for target column: {col}")
            
            encoder = data_loader.encoders[col]
            if not hasattr(encoder, "classes_"):
                raise AttributeError(f"Encoder for '{col}' missing 'classes_' attribute")
            
            sizes.append(len(encoder.classes_))
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting output size for '{col}': {e}")
            raise

    return sizes


def calculate_task_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    target_cols: List[str],
    multi_label_cols: set,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate detailed metrics (F1, precision, recall) for each task.
    
    Handles both single-label and multi-label classification tasks.
    For single-label: uses argmax for predictions.
    For multi-label: uses sigmoid with 0.5 threshold.
    
    Args:
        model: Trained model in eval mode
        data_loader: DataLoader with validation/test data
        target_cols: List of target column names
        multi_label_cols: Set of multi-label column names
        device: Torch device (CPU or CUDA)
    
    Returns:
        Dictionary with structure:
        {
            'task_name': {
                'f1_macro': float,
                'f1_weighted': float,
                'precision_macro': float,
                'recall_macro': float
            }
        }
    """
    model.eval()

    # Store predictions and ground truth
    all_preds = {col: [] for col in target_cols}
    all_targets = {col: [] for col in target_cols}

    with torch.no_grad():
        for batch in data_loader:
            drug = batch["drug"].to(device)
            genalle = batch["genalle"].to(device)
            gene = batch["gene"].to(device)
            allele = batch["allele"].to(device)

            outputs = model(drug, genalle, gene, allele)

            for col in target_cols:
                true = batch[col].to(device)
                pred = outputs[col]

                if col in multi_label_cols:
                    # Multi-label: apply sigmoid and threshold
                    probs = torch.sigmoid(pred)
                    predicted = (probs > 0.5).float()
                else:
                    # Single-label: argmax
                    predicted = torch.argmax(pred, dim=1)

                all_preds[col].append(predicted.cpu())
                all_targets[col].append(true.cpu())

    # Calculate metrics per task
    metrics = {}
    for col in target_cols:
        preds = torch.cat(all_preds[col]).numpy()
        targets = torch.cat(all_targets[col]).numpy()

        if col in multi_label_cols:
            # Multi-label: use 'samples' average
            metrics[col] = {
                "f1_samples": f1_score(targets, preds, average="samples", zero_division=0),
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
                "precision_samples": precision_score(
                    targets, preds, average="samples", zero_division=0
                ),
                "recall_samples": recall_score(
                    targets, preds, average="samples", zero_division=0
                ),
            }
        else:
            # Single-label
            metrics[col] = {
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
                "precision_macro": precision_score(
                    targets, preds, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    targets, preds, average="macro", zero_division=0
                ),
            }

    return metrics


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
    params = get_optuna_params(trial, model_name)

    # 2. Create datasets and dataloaders
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # 3. Create model
    input_dims = get_input_dims(fitted_data_loader)
    n_targets_list = get_output_sizes(fitted_data_loader, target_cols)
    target_dims = {
        col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)
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
    )
    model = model.to(device)

    # 4. Define optimizer and loss functions
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-5),
    )

    criterions = []
    task3_name = "effect_type"

    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        elif col == task3_name and class_weights_task3 is not None:
            criterions.append(
                nn.CrossEntropyLoss(
                    label_smoothing=LABEL_SMOOTHING, weight=class_weights_task3
                )
            )
            if trial.number == 0:
                logger.info(f"Applying class weighting with label smoothing for '{task3_name}'")
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING))

    criterions.append(optimizer)

    # 5. Train model
    best_loss, best_accuracies_list = train_model(
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=EPOCH,
        patience=PATIENCE,
        target_cols=target_cols,
        scheduler=None,
        params_to_txt=params,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        progress_bar=False,
        model_name=model_name,
        use_weighted_loss=False,
        task_priorities=None, # type: ignore
        return_per_task_losses=False,
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
        t
        for t in study.trials
        if t.values is not None and all(isinstance(v, (int, float)) for v in t.values)
    ]

    if not valid_trials:
        logger.warning("No valid trials found in study")
        return []

    # For multi-objective, return Pareto front trials
    if len(valid_trials[0].values) > 1:
        # Already ordered by Pareto dominance
        return valid_trials[:n]
    else:
        # For single-objective, sort by value
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

    best_trial = top_5_trials[0]

    # Save JSON
    output = {
        "model": model_name,
        "optimization_type": "multi_objective" if is_multi_objective else "single_objective",
        "best_trial": {
            "number": best_trial.number,
            "values": list(best_trial.values) if best_trial.values else [],
            "params": best_trial.params,
            "user_attrs": dict(best_trial.user_attrs),
        },
        "top5": [
            {
                "number": t.number,
                "values": list(t.values) if t.values else [],
                "params": t.params,
                "critical_f1": t.user_attrs.get("critical_task_f1", 0.0),
            }
            for t in top_5_trials
        ],
        "n_trials": len(study.trials),
        "clinical_priorities": CLINICAL_PRIORITIES,
    }

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
            f.write(" -----------------------------------------------------------|\n")
            f.write(f"|  OPTIMIZATION REPORT: {model_name:^30} |\n")
            f.write(" -----------------------------------------------------------|\n\n")

            if is_multi_objective:
                f.write("Optimization Type: Multi-Objective\n")
                f.write("   - Objective 1: Minimize Loss\n")
                f.write(
                    f"   - Objective 2: Maximize F1 ({best_trial.user_attrs.get('critical_task_name', 'N/A')})\n\n"
                )
            else:
                f.write("Optimization Type: Single-Objective (Loss)\n\n")

            f.write(f"Best Trial (#{best_trial.number})\n")
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
            for key, value in best_trial.params.items():
                f.write(f"   {key:20s}: {value}\n")
            f.write("\n")

            # Top 5
            f.write("Top 5 Trials\n")
            f.write("-" * 60 + "\n\n")
            for i, t in enumerate(top_5_trials, 1):
                f.write(f"{i}. Trial #{t.number}\n")
                if t.values:
                    if is_multi_objective:
                        f.write(f"   Loss: {t.values[0]:.5f}, F1: {-t.values[1]:.5f}\n")
                    else:
                        f.write(f"   Loss: {t.values[0]:.5f}\n")
                f.write(f"   Params: {t.params}\n\n")

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

    data_loader = PGenDataProcess()
    df = data_loader.load_data(
        csvfiles,   # type: ignore
        cols_from_config,
        targets_from_config,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=["effect_type"],
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
    class_weights_task3 = None
    task3_name = "effect_type"

    if task3_name in target_cols:
        try:
            encoder_task3 = data_loader.encoders[task3_name]
            counts = train_processed_df[task3_name].value_counts().sort_index()
            all_counts = torch.zeros(len(encoder_task3.classes_))
            for class_id, count in counts.items():
                all_counts[int(class_id)] = count    # type: ignore

            weights = 1.0 / torch.log(all_counts + CLASS_WEIGHT_LOG_SMOOTHING)
            weights = weights / weights.mean()
            class_weights_task3 = weights.to(device)
            logger.info(
                f"Class weights calculated for '{task3_name}': "
                f"Min={weights.min():.3f}, Max={weights.max():.3f}, Mean={weights.mean():.3f}"
            )
        except Exception as e:
            logger.warning(f"Error calculating class weights: {e}")

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

    # Create study based on mode
    if use_multi_objective:
        pruner = None
        logger.info("Pruning disabled (incompatible with multi-objective)")
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name=f"optuna_{model_name}_{datetime_study}",
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

    study.optimize(optuna_func, n_trials=n_trials)
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
            if use_multi_objective and len(study.best_trials) > 1:
                fig_pareto = plot_pareto_front(study)
                pareto_filename = optuna_results / f"pareto_{model_name}_{datetime_study}.html"
                fig_pareto.write_html(str(pareto_filename))
                logger.info(f"Pareto front saved: {pareto_filename}")
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
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETED")
    print("=" * 60)
    if use_multi_objective:
        print(f"Best Trial (Pareto-Optimal): #{best_trial.number}")
        print(f"   Loss:           {best_loss:.5f}")
        print(f"   F1 (Critical):  {best_f1:.5f}")
    else:
        print(f"Best Trial: #{best_trial.number}")
        print(f"   Loss:           {best_loss:.5f}")

    print(f"\nReports saved in: {results_dir}")
    print("=" * 60 + "\n")

    return best_params, best_loss, results_file_json, normalized_loss
