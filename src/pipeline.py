# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.

import logging
import torch
import joblib
from torch.utils.data import DataLoader
from pathlib import Path

from src.cfg.manager import get_model_config, DIRS, MULTI_LABEL_COLS
from src.data_handler import load_dataset, PGenProcessor, PGenDataset
from src.modeling import create_model
from src.losses import MultiTaskUncertaintyLoss
from src.trainer import PGenTrainer
from src.utils.io_utils import save_json

logger = logging.getLogger(__name__)

def train_pipeline(model_name: str, csv_path: str|Path, epochs: int = 50):
    """
    Main training workflow.
    1. Config & Data Loading
    2. Preprocessing & Dataset Creation
    3. Model Initialization (via Factory)
    4. Training Loop
    5. Artifact Saving
    """
    # 1. Configuration
    cfg = get_model_config(model_name)
    params = cfg["params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Starting pipeline for model: {model_name}")
    logger.info(f"Device: {device}")

    # 2. Data Loading
    cols_to_load = list(set(cfg["features"] + cfg["targets"]))
    df = load_dataset(csv_path, cols_to_load, stratify_col=cfg.get("stratify_col"))
    
    # Data Splitting
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=711, 
        stratify=df["_stratify"] if "_stratify" in df.columns else None
    )
    
    # 3. Preprocessing
    processor = PGenProcessor(cfg["features"], cfg["targets"], list(MULTI_LABEL_COLS))
    processor.fit(train_df)
    
    train_ds = PGenDataset(processor.transform(train_df), cfg["features"], cfg["targets"], MULTI_LABEL_COLS)
    val_ds = PGenDataset(processor.transform(val_df), cfg["features"], cfg["targets"], MULTI_LABEL_COLS)
    
    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=params["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    # 4. Model Setup
    dims = {col: len(enc.classes_) for col, enc in processor.encoders.items()}
    n_features = {k: v for k,v in dims.items() if k in cfg["features"]}
    target_dims = {k: v for k,v in dims.items() if k in cfg["targets"]}
    
    # Use Factory Pattern
    model = create_model(model_name, n_features, target_dims, params).to(device)
    
    # Uncertainty Loss (Optional)
    unc_module = None
    if params.get("use_uncertainty_loss"):
        logger.info("Using Multi-Task Uncertainty Loss.")
        unc_module = MultiTaskUncertaintyLoss(cfg["targets"]).to(device)
        
    # Optimizer & Scheduler
    trainable_params = list(model.parameters()) + (list(unc_module.parameters()) if unc_module else [])
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=params["learning_rate"], 
        weight_decay=params["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)
    
    # 5. Training Execution
    trainer = PGenTrainer(
        model, optimizer, scheduler, device, 
        cfg["targets"], MULTI_LABEL_COLS, params, unc_module
    )
    
    trainer.fit(train_loader, val_loader, epochs=epochs, patience=params["early_stopping_patience"])
    
    # 6. Artifacts
    joblib.dump(processor.encoders, DIRS["encoders"] / f"encoders_{model_name}.pkl")
    save_json(cfg, DIRS["reports"] / f"{model_name}_config.json")
    
    logger.info("Pipeline completed successfully.")

