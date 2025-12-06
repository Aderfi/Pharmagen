# Pharmagen - Training Pipeline
# Orchestrates data loading, model setup, and training loop.

import logging
import torch
import joblib
from pathlib import Path
from typing import Optional

from src.cfg.manager import get_model_config, DIRS, MULTI_LABEL_COLS
from src.data_handler import load_dataset, PGenProcessor
from src.losses import MultiTaskUncertaintyLoss
from src.trainer import PGenTrainer
from src.utils.io_utils import save_json
from src.factories import create_model_instance, create_data_loaders

logger = logging.getLogger(__name__)

def train_pipeline(model_name: str, csv_path: str|Path, epochs: int = 50):
    """
    Main training workflow.
    1. Config & Data Loading
    2. Preprocessing & Dataset Creation (via Factory)
    3. Model Initialization (via Factory)
    4. Training Loop
    5. Artifact Saving
    """
    # 1. Configuration
    cfg = get_model_config(model_name)
    params = cfg["params"]
    model_type = cfg.get("model_type", "tabular")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Starting pipeline for model: {model_name} (Type: {model_type})")
    logger.info(f"Device: {device}")

    # 2. Data Loading (Raw DataFrame)
    # Both modes use a CSV/TSV as the "Index" or "Data Source"
    cols_to_load = list(set(cfg["features"] + cfg["targets"]))
    # Note: For Graph mode, 'features' might be empty or contain drug/gene col names
    # load_dataset is robust enough to handle basic loading
    df = load_dataset(csv_path, cols_to_load, stratify_col=cfg.get("stratify_col"))
    
    # Data Splitting
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=711, 
        stratify=df["_stratify"] if "_stratify" in df.columns else None
    )
    
    # 3. Preprocessing & Loaders (Via Factory)
    processor = None
    dims = {}
    
    if model_type == "tabular":
        logger.info(" fitting Tabular Processor...")
        processor = PGenProcessor(cfg["features"], cfg["targets"], list(MULTI_LABEL_COLS))
        processor.fit(train_df)
        
        # Dimensions for Tabular Model
        dims = {col: len(enc.classes_) for col, enc in processor.encoders.items()}
        dims = {
            "n_features": {k: v for k,v in dims.items() if k in cfg["features"]},
            "target_dims": {k: v for k,v in dims.items() if k in cfg["targets"]}
        }
    else:
        # Dimensions for Graph Model
        # Default dimensions or read from params
        # In a real scenario, we might verify target dimensions from data
        dims = {
            "target_dims": {col: 1 for col in cfg["targets"]}, # Simplified: Reg/Binary=1. Multiclass needs logic.
            # If Multiclass, we need to calculate classes from df
            "drug_node_features": 9, # Default RDKit
            "gene_input_dim": 768 # DNABERT default
        }
        # Correct target dims for multiclass if needed
        # (This logic is a bit simplified for Refactor, but follows the pattern)
    
    train_loader, val_loader = create_data_loaders(
        config=cfg,
        df_train=train_df,
        df_val=val_df,
        processor=processor
    )
    
    # 4. Model Setup (Via Factory)
    model = create_model_instance(cfg, dims).to(device)
    
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
    if processor:
        joblib.dump(processor.encoders, DIRS["encoders"] / f"encoders_{model_name}.pkl")
        
    save_json(cfg, DIRS["reports"] / f"{model_name}_config.json")
    
    logger.info("Pipeline completed successfully.")

