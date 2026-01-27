"""
factories.py

Centralizes the creation of Models, Datasets, and DataLoaders.
Implements the Abstract Factory pattern to switch between strategies.

Adheres to: SOLID (Open/Closed), Zen of Python (Explicit is better than implicit).
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from torch.utils.data import DataLoader

from src.cfg.manager import DIRS
from src.modeling import create_model as create_deepfm
from src.data_handler import PGenDataset

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model_instance(
    config: Dict[str, Any], 
    dims: Dict[str, Any]
) -> torch.nn.Module:
    """
    Creates a model instance based on the configuration.
    
    Args:
        config: Model configuration dictionary.
        dims: Dictionary containing dimension info (n_features, target_dims, etc.)
    """
    params = config["params"]
    
    # Tabular DeepFM
    # dims expected: {'n_features': {...}, 'target_dims': {...}}
    return create_deepfm(
        model_name=config.get("name", "DeepFM"), 
        n_features=dims.get("n_features", {}),
        target_dims=dims.get("target_dims", {}),
        params=params
    )
        

# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def create_data_loaders(
    config: Dict[str, Any],
    df_train,
    df_val,
    processor=None, 
    drug_list_path: Optional[str] = None,
    ref_genome_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates Train and Val DataLoaders.
    """
    params = config["params"]
    batch_size = params["batch_size"]
    
    if not processor:
        raise ValueError("Processor required for Tabular data loaders.")
        
    # Transform DataFrames
    train_data = processor.transform(df_train)
    val_data = processor.transform(df_val)
    
    # Create PGenDatasets
    train_ds = PGenDataset(train_data, config["features"], config["targets"], processor.multi_label_cols)
    val_ds = PGenDataset(val_data, config["features"], config["targets"], processor.multi_label_cols)
    
    # Standard Loaders
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    )
