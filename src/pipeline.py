"""
pipeline.py

Orchestrates the End-to-End Training Pipeline:
Config -> Data Loading -> Preprocessing -> Training -> Artifact Saving.
"""

from dataclasses import asdict
import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import torch

from src.cfg.manager import DIRS, MULTI_LABEL_COLS, get_model_config
from src.data.data_handler import (
    DataConfig,
    PGenProcessor,
    load_and_clean_dataset,
)
from src.factories import create_data_loaders, create_model_instance
from src.interface.io import json_serial_adapter
from src.model.engine.trainer import PGenTrainer, TrainerConfig
from src.model.metrics.losses import MultiTaskUncertaintyLoss

logger = logging.getLogger(__name__)


def _parse_configs(
    model_name: str,
    csv_path: Path,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
):
    """
    Parses raw TOML configuration into structured Dataclasses.

    Args:
        model_name (str): Key in models.toml.
        csv_path (Path): Path to dataset.
        epochs_override (int): Overwrite epoch count.
        batch_size_override (int): Overwrite batch size.

    Returns:
        tuple: (DataConfig, TrainerConfig, FullConfigDict)
    """
    raw_cfg = get_model_config(model_name)
    train_params = raw_cfg.get("training", {})
    loss_params = raw_cfg.get("loss", {})

    # Apply batch size override directly to raw_cfg before factory uses it
    if batch_size_override:
        train_params["batch_size"] = batch_size_override
        # Important: Update the raw dict so create_data_loaders sees it too
        raw_cfg["training"]["batch_size"] = batch_size_override

    # 1. Data Configuration
    data_config = DataConfig(
        dataset_path=csv_path,
        feature_cols=raw_cfg["data"]["features"],
        target_cols=raw_cfg["data"]["targets"],
        multi_label_cols=list(MULTI_LABEL_COLS),
        stratify_col=raw_cfg["data"].get("stratify_col"),
        num_workers=4,
        pin_memory=True,
    )

    # 2. Trainer Configuration
    trainer_config = TrainerConfig(
        n_epochs=epochs_override if epochs_override else train_params.get("epochs", 50),
        patience=train_params.get("early_stopping_patience", 10),
        learning_rate=train_params.get("learning_rate", 1e-3),
        weight_decay=train_params.get("weight_decay", 1e-4),
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True,
        ml_loss_type=loss_params.get("loss_multilabel", "asymmetric"),
        mc_loss_type=loss_params.get("loss_singlelabel", "adaptive_focal"),
        label_smoothing=loss_params.get("label_smoothing", 0.0),
    )

    return data_config, trainer_config, raw_cfg


def train_pipeline(
    model_name: str, csv_path: str | Path, epochs: int = 50, batch_size: int | None = None
):
    """
    Orchestrates the End-to-End Training Pipeline.

    Process:
    1. Loads Config & Data
    2. Splits Data (Train/Val)
    3. Fits Encoders & Preprocesses
    4. Builds Model via Factory
    5. Sets up Loss/Optimizer
    6. Runs Trainer Loop
    7. Saves Artifacts (Model + Encoders)

    Args:
        model_name (str): Model identifier.
        csv_path (str | Path): Path to raw CSV.
        epochs (int): Number of epochs.
        batch_size (int, optional): Override config batch size.
    """
    csv_path = Path(csv_path)
    logger.info("--- Starting Pipeline: %s ---", model_name)

    # 1. Config Setup
    data_cfg, trainer_cfg, full_config = _parse_configs(model_name, csv_path, epochs, batch_size)
    logger.info("Device: %s", trainer_cfg.device)

    # 2. Data Loading & Processing
    df = load_and_clean_dataset(data_cfg)

    # Split
    stratify = df["_stratify"] if "_stratify" in df.columns else None
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=stratify)

    # Process
    processor = PGenProcessor(full_config["data"], list(MULTI_LABEL_COLS))
    processor.fit(train_df)

    # 3. Create DataLoaders (Factory)
    train_loader, val_loader = create_data_loaders(
        config=full_config, df_train=train_df, df_val=val_df, processor=processor
    )

    # 4. Model Setup (Factory)
    # Dimensions Extraction for Model
    all_dims = {col: len(enc.classes_) for col, enc in processor.encoders.items()}
    feat_dims = {k: v for k, v in all_dims.items() if k in data_cfg.feature_cols}
    target_dims = {k: v for k, v in all_dims.items() if k in data_cfg.target_cols}

    dims_payload = {"n_features": feat_dims, "target_dims": target_dims}

    model = create_model_instance(full_config, dims_payload)

    # 5. Loss & Uncertainty
    unc_module = None
    loss_params = full_config.get("loss", {})
    if loss_params.get("use_uncertainty_loss", False):
        logger.info("Initializing Multi-Task Uncertainty Loss...")
        unc_module = MultiTaskUncertaintyLoss(list(target_dims.keys()))

    # Optimization
    params_to_optimize = list(model.parameters()) + (
        list(unc_module.parameters()) if unc_module else []
    )
    optimizer = torch.optim.AdamW(
        params_to_optimize, lr=trainer_cfg.learning_rate, weight_decay=trainer_cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 6. Training Loop
    trainer = PGenTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_cfg,
        target_cols=data_cfg.target_cols,
        multi_label_cols=set(data_cfg.multi_label_cols),
        model_name=model_name,
        uncertainty_module=unc_module,
    )

    best_loss = trainer.fit(train_loader, val_loader)

    # 7. Artifact Preservation
    logger.info("Saving artifacts...")

    # A. Encoders (Critical for Inference)
    encoder_path = DIRS["encoders"] / f"encoders_{model_name}.pkl"
    joblib.dump(processor.encoders, encoder_path)

    # B. Configuration Snapshot
    config_snapshot = {
        "full_config": full_config,
        "trainer_config": asdict(trainer_cfg),
        "final_loss": best_loss,
    }
    with open(DIRS["reports"] / f"{model_name}_final_config.json", "w") as f:
        json.dump(config_snapshot, f, indent=4, default=json_serial_adapter)

    logger.info("Pipeline Finished. Model saved at %s", DIRS["models"])


def train_kfold_pipeline(
    model_name: str,
    csv_path: str | Path,
    k_folds: int = 5,
    epochs: int = 50,
    batch_size: int | None = None,
):
    """
    Executes k-Fold Cross Validation Pipeline.

    Args:
        model_name (str): Model ID.
        csv_path (Path): Path to data.
        k_folds (int): Number of folds (default: 5).
        epochs (int): Epochs per fold.
        batch_size (int): Batch size override.
    """
    csv_path = Path(csv_path)
    logger.info("--- Starting %d-Fold Cross-Validation: %s ---", k_folds, model_name)

    # 1. Config & Data Loading
    data_cfg, trainer_cfg, full_config = _parse_configs(model_name, csv_path, epochs, batch_size)
    df = load_and_clean_dataset(data_cfg)

    # 2. Prepare CV Strategy
    stratify_col = df["_stratify"] if "_stratify" in df.columns else None

    if stratify_col is not None:
        # Check minimal class count for stratification
        if df[stratify_col].value_counts().min() < k_folds:
            logger.warning("Some strata have fewer samples than k_folds. Falling back to KFold.")
            skf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            splits = list(skf.split(df))
        else:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            splits = list(skf.split(df, df[stratify_col]))
    else:
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = list(kf.split(df))

    fold_metrics = []
    best_overall_loss = float("inf")

    # Pre-fit Processor on FULL dataset for consistent vocab
    processor = PGenProcessor(full_config["data"], list(MULTI_LABEL_COLS))
    processor.fit(df)

    # Save global encoders
    encoder_path = DIRS["encoders"] / f"encoders_{model_name}.pkl"
    joblib.dump(processor.encoders, encoder_path)

    # 3. Iterate Folds
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info("=== Fold %d/%d ===", fold_idx + 1, k_folds)

        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Create Loaders
        train_loader, val_loader = create_data_loaders(
            config=full_config, df_train=train_df, df_val=val_df, processor=processor
        )

        # Init Model
        all_dims = {col: len(enc.classes_) for col, enc in processor.encoders.items()}
        feat_dims = {k: v for k, v in all_dims.items() if k in data_cfg.feature_cols}
        target_dims = {k: v for k, v in all_dims.items() if k in data_cfg.target_cols}
        dims_payload = {"n_features": feat_dims, "target_dims": target_dims}

        model = create_model_instance(full_config, dims_payload)

        # Loss & Opt
        unc_module = None
        loss_params = full_config.get("loss", {})
        if loss_params.get("use_uncertainty_loss", False):
            unc_module = MultiTaskUncertaintyLoss(list(target_dims.keys()))

        params_to_optimize = list(model.parameters()) + (
            list(unc_module.parameters()) if unc_module else []
        )
        optimizer = torch.optim.AdamW(
            params_to_optimize, lr=trainer_cfg.learning_rate, weight_decay=trainer_cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # Trainer
        trainer = PGenTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_cfg,
            target_cols=data_cfg.target_cols,
            multi_label_cols=set(data_cfg.multi_label_cols),
            model_name=f"{model_name}_fold{fold_idx}",
            uncertainty_module=unc_module,
        )

        final_loss = trainer.fit(train_loader, val_loader)
        fold_metrics.append(final_loss)

        # Save best model logic
        if final_loss < best_overall_loss:
            best_overall_loss = final_loss
            logger.info("New best fold found (Loss: %.4f). Saving as main model.", final_loss)
            fold_best_path = DIRS["models"] / "model_best.pt"
            prod_path = DIRS["models"] / f"pmodel_{model_name}.pt"
            if fold_best_path.exists():
                state = torch.load(fold_best_path)
                torch.save(state, prod_path)

    # 4. Report Results
    mean_loss = np.mean(fold_metrics)
    std_loss = np.std(fold_metrics)

    logger.info("--- Cross-Validation Results ---")
    logger.info("Loss per fold: %s", [f"{x:.4f}" for x in fold_metrics])
    logger.info("Mean Loss: %.4f ± %.4f", mean_loss, std_loss)

    report = {
        "k_folds": k_folds,
        "fold_metrics": fold_metrics,
        "mean_loss": mean_loss,
        "std_loss": std_loss,
        "best_loss": best_overall_loss,
    }
    with open(DIRS["reports"] / f"{model_name}_cv_results.json", "w") as f:
        json.dump(report, f, indent=4)
