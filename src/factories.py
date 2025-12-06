"""
factories.py

Centralizes the creation of Models, Datasets, and DataLoaders.
Implements the Abstract Factory pattern to switch between Tabular and Graph strategies.

Adheres to: SOLID (Open/Closed), Zen of Python (Explicit is better than implicit).
"""

import logging
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

from src.cfg.manager import DIRS, _PATHS_CFG, _resolve
from src.modeling import create_model as create_deepfm
from src.modeling_graph import PGenGraphModel
from src.data_handler import PGenDataset
from src.datasets import DrugGraphDataset, GeneEmbeddingDataset
from src.datasets.interactions import GraphInteractionDataset, graph_interaction_collator

logger = logging.getLogger(__name__)

# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model_instance(
    config: Dict[str, Any], 
    dims: Dict[str, Any]
) -> torch.nn.Module:
    """
    Creates a model instance based on the configuration 'model_type'.
    
    Args:
        config: Model configuration dictionary.
        dims: Dictionary containing dimension info (n_features, target_dims, etc.)
              Expected keys differ by model type.
    """
    model_type = config.get("model_type", "tabular")
    params = config["params"]
    
    if model_type == "tabular":
        # Tabular DeepFM
        # dims expected: {'n_features': {...}, 'target_dims': {...}}
        return create_deepfm(
            model_name=config.get("name", "DeepFM"), # Name is arbitrary here
            n_features=dims.get("n_features", {}),
            target_dims=dims.get("target_dims", {}),
            n_graph_features=0, # Not used in tabular
            params=params
        )
        
    elif model_type == "graph":
        # Graph PGenModel
        # dims expected: {'drug_node_features': int, 'gene_input_dim': int, 'target_dims': {...}}
        return PGenGraphModel(
            drug_node_features=dims.get("drug_node_features", 9), # Default RDKit atom features?
            gene_input_dim=dims.get("gene_input_dim", 768), # Default DNABERT
            target_dims=dims.get("target_dims", {}),
            hidden_dim=params.get("hidden_dim", 256),
            embedding_dim=params.get("embedding_dim", 128),
            n_gnn_layers=params.get("n_gnn_layers", 3),
            dropout_rate=params.get("dropout_rate", 0.2)
        )
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# =============================================================================
# DATA LOADER FACTORY
# =============================================================================

def create_data_loaders(
    config: Dict[str, Any],
    df_train,
    df_val,
    processor=None, # Only for tabular
    drug_list_path: Optional[str] = None,
    ref_genome_path: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates Train and Val DataLoaders.
    """
    model_type = config.get("model_type", "tabular")
    params = config["params"]
    batch_size = params["batch_size"]
    
    if model_type == "tabular":
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
        
    elif model_type == "graph":
        # 1. Setup Sub-Datasets (Shared)
        if not drug_list_path:
             # Fallback or error
             drug_list_path = str(DIRS["dicts"] / "drug_list.txt")
        
        if not ref_genome_path:
            # Try to get from paths.toml loaded in manager
            ref_genome_path = str(_resolve(_PATHS_CFG["genome_references"]["ref_genome_fasta"]))

        logger.info("Initializing Graph Sub-Datasets...")
        # Cached Drug Graphs
        drug_ds = DrugGraphDataset(root=str(DIRS["data"] / "graphs"), drug_list_file=drug_list_path)
        
        # Gene Embeddings (On-the-fly)
        # We pass a dummy list to init, as InteractionDataset calls the method directly.
        gene_ds = GeneEmbeddingDataset(
            data_source=[], 
            ref_genome_path=ref_genome_path, 
            gtf_path="", # Not used yet
            model_name=params.get("dna_model", "zhihan1996/DNABERT-2-117M")
        )
        
        # 2. Create Interaction Datasets
        target_cols = config["targets"]
        
        train_ds = GraphInteractionDataset(df_train, drug_ds, gene_ds, target_cols)
        val_ds = GraphInteractionDataset(df_val, drug_ds, gene_ds, target_cols)
        
        # 3. PyG DataLoaders (Custom Collator)
        # PyG DataLoader handles Batch object creation if list of Data is returned.
        # But we return a dict/tuple. We use our custom collator.
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=graph_interaction_collator),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=graph_interaction_collator)
        )
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
