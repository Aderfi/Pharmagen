import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import joblib
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.cfg.manager import DIRS, MULTI_LABEL_COLS, get_model_config
from src.data_handler import (
    DataConfig,
    PGenDataset,
    PGenProcessor,
    load_and_clean_dataset,
)
from src.losses import MultiTaskUncertaintyLoss
from src.modeling import ModelConfig, PharmagenDeepFM
from src.trainer import PGenTrainer, TrainerConfig

logger = logging.getLogger("Pharmagen.Pipeline")


def _parse_configs(model_name: str, csv_path: Path, epochs_override: int | None = None):
    """
    Parses raw dictionary config into strict Dataclasses.
    """
    raw_cfg = get_model_config(model_name)
    params = raw_cfg.get("params", {})

    # 1. Data Configuration
    data_config = DataConfig(
        dataset_path=csv_path,
        feature_cols=raw_cfg["features"],
        target_cols=raw_cfg["targets"],
        multi_label_cols=list(MULTI_LABEL_COLS),
        stratify_col=raw_cfg.get("stratify_col"),
        num_workers=4, # Safe for single run on Linux
        pin_memory=True
    )

    # 2. Trainer Configuration
    trainer_config = TrainerConfig(
        n_epochs=epochs_override if epochs_override else params.get("epochs", 50),
        patience=params.get("patience", 10),
        learning_rate=params.get("learning_rate", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True,
        ml_loss_type=params.get("ml_loss_type", "bce"),
        mc_loss_type=params.get("mc_loss_type", "focal"),
        label_smoothing=params.get("label_smoothing", 0.0)
    )

    # 3. Model Config Partial (Dims will be filled after data loading)
    model_config_partial = {
        "embedding_dim": params.get("embedding_dim", 64),
        "embedding_dropout": params.get("embedding_dropout", 0.1),
        "hidden_dim": params.get("hidden_dim", 256),
        "n_layers": params.get("n_layers", 3),
        "dropout_rate": params.get("dropout_rate", 0.2),
        "activation": params.get("activation", "gelu"),
        "use_transformer": params.get("use_transformer", True),
        "attn_dim_feedforward": params.get("attn_dim_feedforward", 512),
        "attn_heads": params.get("attn_heads", 4),
        "num_attn_layers": params.get("num_attn_layers", 2),
        "fm_hidden_dim": params.get("fm_hidden_dim", 64)
    }

    return data_config, trainer_config, model_config_partial, params

def train_pipeline(model_name: str, csv_path: str | Path, epochs: int = 50):
    """
    Orchestrates the End-to-End Training Pipeline.
    """
    csv_path = Path(csv_path)
    logger.info(f"--- Starting Pipeline: {model_name} ---")

    # 1. Config Setup
    data_cfg, trainer_cfg, model_cfg_dict, raw_params = _parse_configs(model_name, csv_path, epochs)

    logger.info(f"Device: {trainer_cfg.device}")

    # 2. Data Loading & Processing
    df = load_and_clean_dataset(data_cfg)

    # Split
    stratify = df["_stratify"] if "_stratify" in df.columns else None
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify
    )

    # Process
    processor = PGenProcessor(data_cfg)
    processor.fit(train_df)

    # Dimensions Extraction for Model
    all_dims = {col: len(enc.classes_) for col, enc in processor.encoders.items()}
    feat_dims = {k: v for k, v in all_dims.items() if k in data_cfg.feature_cols}
    target_dims = {k: v for k, v in all_dims.items() if k in data_cfg.target_cols}

    # Create Final Model Config
    model_cfg = ModelConfig(
        n_features=feat_dims,
        target_dims=target_dims,
        **model_cfg_dict
    )

    # Datasets -> Tensors
    train_dataset = PGenDataset(processor.transform(train_df))
    val_dataset = PGenDataset(processor.transform(val_df))

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=raw_params.get("batch_size", 128),
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=True # Optimization for Linux/Debian
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=raw_params.get("batch_size", 128),
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        persistent_workers=True
    )

    # 3. Model & Components Setup
    # -------------------------------------------------------------------------
    model = PharmagenDeepFM(model_cfg)

    # Uncertainty Module (Optional)
    unc_module = None
    if raw_params.get("use_uncertainty_loss", False):
        logger.info("Initializing Multi-Task Uncertainty Loss...")
        unc_module = MultiTaskUncertaintyLoss(list(target_dims.keys()))

    # Optimization
    params_to_optimize = list(model.parameters()) + (list(unc_module.parameters()) if unc_module else [])
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=trainer_cfg.learning_rate,
        weight_decay=trainer_cfg.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 4. Training Loop
    # -------------------------------------------------------------------------
    trainer = PGenTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=trainer_cfg,
        target_cols=data_cfg.target_cols,
        multi_label_cols=set(data_cfg.multi_label_cols),
        uncertainty_module=unc_module
    )

    best_loss = trainer.fit(train_loader, val_loader)

    # 5. Artifact Preservation
    # -------------------------------------------------------------------------
    logger.info("Saving artifacts...")

    # A. Encoders (Critical for Inference)
    encoder_path = DIRS["encoders"] / f"encoders_{model_name}.pkl"
    joblib.dump(processor.encoders, encoder_path)

    # B. Configuration Snapshot
    config_snapshot = {
        "model_config": asdict(model_cfg),
        "trainer_config": asdict(trainer_cfg),
        "data_config": asdict(data_cfg),
        "final_loss": best_loss
    }
    # Helper to serialize Paths
    def _json_serial(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError (f"Type {type(obj)} not serializable")

    with open(DIRS["reports"] / f"{model_name}_final_config.json", "w") as f:
        json.dump(config_snapshot, f, indent=4, default=_json_serial)

    logger.info(f"Pipeline Finished. Model saved at {DIRS['models']}")
