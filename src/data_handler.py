"""
data_handler.py

Implements efficient data loading, cleaning, and preprocessing pipelines.
Designed to handle heterogeneous biological data (scalars, multi-labels).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset

from src.utils.data_utils import clean_dataset_content

logger = logging.getLogger(__name__)

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for Data Loading and Processing.
    """
    dataset_path: Path
    feature_cols: list[str]
    target_cols: list[str]
    multi_label_cols: list[str]
    stratify_col: str | None = None
    num_workers: int = 4
    pin_memory: bool = True

# =============================================================================
# 1. PREPROCESSING ENGINE
# =============================================================================

class PGenProcessor(BaseEstimator, TransformerMixin):
    """
    Stateful processor that converts Pandas DataFrame -> Dictionary of Tensors.
    
    Decouples Pandas overhead from the training loop by pre-computing
    encodings and transforming data into pure PyTorch tensors before
    Dataset instantiation.
    """
    def __init__(self, config: dict, multi_label_cols: list[str] | None = None):
        self.cfg = config
        self.encoders: dict[str, Any] = {}
        self.multi_label_cols = set(c.lower() for c in (multi_label_cols or []))
        
        # Identify columns
        all_cols = config.get("features", []) + config.get("targets", [])
        self.target_cols = [c.lower() for c in config.get("targets", [])]
        self.scalar_cols = [
            c.lower() for c in all_cols 
            if c.lower() not in self.multi_label_cols and c.lower() not in self.target_cols
        ]
        
        self.unknown_token = "__UNKNOWN__"

    def fit(self, df: pd.DataFrame, y=None):
        logger.info("Fitting feature encoders...")

        # Fit Scalars (LabelEncoding)
        for col in self.scalar_cols + self.target_cols:
            if col not in df.columns:
                continue

            uniques = set(df[col].dropna().unique())
            uniques.add(self.unknown_token)

            enc = LabelEncoder()
            enc.fit(sorted(list(uniques)))
            self.encoders[col] = enc

        # Fit Multi-Labels (Binarizer)
        for col in self.multi_label_cols:
            if col not in df.columns:
                continue

            parsed = df[col].astype(str).str.split("|").map(
                lambda x: [i.strip() for i in x if i.strip()] if isinstance(x, list) else []
            )
            enc = MultiLabelBinarizer()
            enc.fit(parsed)
            self.encoders[col] = enc

        return self

    def transform(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        output_payload = {}

        # 1. Transform Scalars
        for col in self.scalar_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            vals = df[col].fillna(self.unknown_token).to_numpy()
            
            unknown_idx = np.searchsorted(enc.classes_, self.unknown_token) # noqa
            
            valid_mask = np.isin(vals, enc.classes_)
            vals[~valid_mask] = self.unknown_token

            encoded_data = enc.transform(vals)
            output_payload[col] = torch.tensor(encoded_data, dtype=torch.long)

        # 2. Transform Targets
        for col in self.target_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            encoded_data = enc.transform(df[col].to_numpy())
            output_payload[col] = torch.tensor(encoded_data, dtype=torch.long)

        # 3. Transform Multi-Labels
        for col in self.multi_label_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            parsed = df[col].astype(str).str.split("|").map(
                lambda x: [i.strip() for i in x if i.strip()] if isinstance(x, list) else []
            )
            mat = enc.transform(parsed).astype(np.float32)
            output_payload[col] = torch.from_numpy(mat)

        return output_payload

# =============================================================================
# 2. MEMORY-OPTIMIZED DATASET
# =============================================================================

class PGenDataset(Dataset):
    """
    Zero-copy PyTorch Dataset.
    """
    def __init__(
        self, 
        data_payload: dict[str, torch.Tensor],
        features: list[str],
        targets: list[str],
        multi_label_cols: set[str]
    ):
        self.data = data_payload
        self.features = [f.lower() for f in features]
        self.targets = [t.lower() for t in targets]
        self.ml_cols = set(c.lower() for c in multi_label_cols)
        
        if not self.data:
            raise ValueError("Empty data payload provided to Dataset.")
            
        first_key = next(iter(self.data))
        self.length = len(self.data[first_key])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = {
            k: self.data[k][idx] 
            for k in self.features 
            if k in self.data
        }
        
        y = {}
        for k in self.targets:
            if k in self.data:
                val = self.data[k][idx]
                y[k] = val.float() if k in self.ml_cols else val
                
        return x, y

def load_and_clean_dataset(config: DataConfig) -> pd.DataFrame:
    """
    Loads dataset efficiently, validating schema and performing initial cleanup.
    """
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {config.dataset_path}")

    logger.info(f"Loading dataset from {config.dataset_path.name}...")

    try:
        df = pd.read_csv(config.dataset_path, sep='\t' if config.dataset_path.suffix=='.tsv' else ',')
    except ValueError as e:
        logger.error(f"Read error: {e}")
        raise

    # Clean content (strip, lower, etc)
    df = clean_dataset_content(df, config.multi_label_cols)
    
    return df
