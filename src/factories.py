"""
factories.py

Centralizes the creation of Models, Datasets, and DataLoaders.
Implements the Abstract Factory pattern to switch between strategies.
"""

import logging
from typing import Any

import torch
from model.architecture.deep_fm import ModelConfig, PharmagenDeepFM
from torch.utils.data import DataLoader

from src.data.data_handler import PGenDataset

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model_instance(
    config: dict[str, Any],
    dims: dict[str, Any]
) -> torch.nn.Module:
    """
    Creates a model instance based on the configuration dictionary.

    Extracts the 'architecture' section from the config, injects runtime
    dimensions (feature cardinality and target shapes), and instantiates
    the model.

    Args:
        config (Dict): Full configuration dict (must contain 'architecture').
        dims (Dict): Dimension info (keys: 'n_features', 'target_dims').

    Returns:
        torch.nn.Module: Instantiated PGenModel ready for training.

    Example:
        >>> cfg = {'architecture': {'hidden_dim': 128}}
        >>> dims = {'n_features': {'drug': 50}, 'target_dims': {'y': 1}}
        >>> model = create_model_instance(cfg, dims)
    """
    # 1. Extract Architecture Params
    arch_params = config.get("architecture", {}).copy()

    # 2. Inject dimensions (These are dynamic, calculated at runtime)
    arch_params["n_features"] = dims.get("n_features", {})
    arch_params["target_dims"] = dims.get("target_dims", {})

    # 3. Create Configuration Object
    model_cfg = ModelConfig(**arch_params)

    # 4. Instantiate Model
    return PharmagenDeepFM(model_cfg)


# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def create_data_loaders(
    config: dict[str, Any],
    df_train,
    df_val,
    processor=None,
    drug_list_path: str | None = None,
    ref_genome_path: str | None = None
) -> tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for Training and Validation.

    Args:
        config (Dict): Config dict (must contain 'training' and 'data').
        df_train (DataFrame): Training split.
        df_val (DataFrame): Validation split.
        processor (DataProcessor): Processor instance for transformation.

    Returns:
        Tuple[DataLoader, DataLoader]: (Train Loader, Val Loader).

    Raises:
        ValueError: If processor is missing.
    """
    # Extract Training Params
    train_params = config.get("training", {})
    batch_size = train_params.get("batch_size", 128)

    if not processor:
        raise ValueError("Processor required for Tabular data loaders.")

    # Transform DataFrames
    train_data = processor.transform(df_train)
    val_data = processor.transform(df_val)

    # Create PGenDatasets
    data_cfg = config.get("data", {})
    train_ds = PGenDataset(train_data, data_cfg["features"], data_cfg["targets"], processor.multi_label_cols)
    val_ds = PGenDataset(val_data, data_cfg["features"], data_cfg["targets"], processor.multi_label_cols)

    # Standard Loaders
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    )
