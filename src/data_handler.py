# Pharmagen - Data Handler
# Unified Data Loading, Preprocessing, and Dataset definition.
# Adheres to Zen of Python: Sparse is better than dense.

import logging
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split

from src.utils.data_utils import serialize_multilabel

# Optional fuzzy matching
try:
    from rapidfuzz import process
except ImportError:
    process = None

logger = logging.getLogger(__name__)

UNKNOWN_TOKEN = "__UNKNOWN__"

# =============================================================================
# 1. DATA LOADING & CLEANING
# =============================================================================

def load_dataset(
    csv_path: Union[str, Path],
    cols_to_load: List[str],
    multi_label_cols: Optional[List[str]] = None,
    stratify_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Robustly loads, cleans, and standardizes the dataset.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # Normalize column requests
    cols_lower = [c.lower() for c in cols_to_load]
    multi_label_lower = [c.lower() for c in (multi_label_cols or [])]

    logger.info(f"Loading dataset from {path.name}...")
    
    # Load
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.lower().str.strip()

    # Normalize Columns
    df.columns = df.columns.str.lower().str.strip()

    # Clean Content
    for col in df.columns:
        if col in multi_label_lower:
            df[col] = df[col].apply(serialize_multilabel).str.lower()
        else:
            # Single label / Feature cleaning
            df[col] = (
                df[col].fillna(UNKNOWN_TOKEN)
                .astype(str)
                .str.replace(", ", ",")
                .str.replace(r"[,;]+", "|", regex=True)
                .str.strip()
                .str.lower()
            )

    # Stratification Helper
    if stratify_col:
        s_cols = [c.strip() for c in stratify_col.lower().split(",")]
        valid_s_cols = [c for c in s_cols if c in df.columns]
        
        if valid_s_cols:
            df["_stratify"] = df[valid_s_cols].astype(str).agg("|".join, axis=1)
            # Filter singletons to allow splitting
            counts = df["_stratify"].value_counts()
            df = df[df["_stratify"].isin(counts[counts > 1].index)].reset_index(drop=True)
        else:
            df["_stratify"] = "default"
            
    return df

# =============================================================================
# 2. PREPROCESSING (TRANSFORMER)
# =============================================================================

class PGenProcessor(BaseEstimator, TransformerMixin):
    """
    Handles encoding of categorical and multi-label features.
    Wraps LabelEncoder and MultiLabelBinarizer.
    """
    def __init__(self, feature_cols: List[str], target_cols: List[str], multi_label_cols: List[str]):
        self.feature_cols = [c.lower() for c in feature_cols]
        self.target_cols = [c.lower() for c in target_cols]
        self.multi_label_cols = set(c.lower() for c in multi_label_cols)
        self.encoders: Dict[str, Any] = {}
        self.cols_to_process = set(self.feature_cols + self.target_cols)

    def fit(self, df: pd.DataFrame, y=None):
        logger.info("Fitting encoders...")
        for col in self.cols_to_process:
            if col not in df.columns: continue

            series = df[col]
            if col in self.multi_label_cols:
                # Split strings to lists for MLB
                parsed = series.apply(lambda x: x.split("|") if x else [])
                enc = MultiLabelBinarizer()
                enc.fit(parsed)
                self.encoders[col] = enc
            else:
                # Single label
                uniques = sorted(list(set(series.unique()) | {UNKNOWN_TOKEN}))
                enc = LabelEncoder()
                enc.fit(uniques)
                self.encoders[col] = enc
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.encoders:
            raise RuntimeError("Processor not fitted.")
        
        df_out = df.copy()
        for col, enc in self.encoders.items():
            if col not in df_out.columns: continue

            if isinstance(enc, MultiLabelBinarizer):
                # String -> List -> Multi-Hot Matrix
                parsed = df_out[col].apply(lambda x: x.split("|") if isinstance(x, str) and x else [])
                # Store as object (numpy array inside cell) to keep DataFrame structure
                encoded = list(enc.transform(parsed))
                df_out[col] = pd.Series(encoded, index=df_out.index)
            else:
                # String -> Int
                # Handle unknown values safely
                vals = df_out[col].astype(str).to_numpy()
                mask_unknown = ~np.isin(vals, enc.classes_)
                if mask_unknown.any():
                    vals[mask_unknown] = UNKNOWN_TOKEN
                df_out[col] = enc.transform(vals)
        
        return df_out

# =============================================================================
# 3. PYTORCH DATASET
# =============================================================================

class PGenDataset(Dataset):
    """
    Optimized Dataset using contiguous memory arrays for speed.
    Separates scalar features (LongTensor) from dense/multi-hot features (FloatTensor).
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_cols: List[str],
        multi_label_cols: Set[str]
    ):
        self.scalar_data = {}
        self.dense_data = {}
        self.length = len(df)
        
        cols = [c.lower() for c in (feature_cols + target_cols) if c in df.columns]
        multi_label_cols = {c.lower() for c in multi_label_cols}

        for col in cols:
            series = df[col]
            if col in multi_label_cols:
                # List of arrays -> 2D Matrix -> Float32
                # Assumes series contains numpy arrays from PGenProcessor
                matrix = np.stack(series.tolist()).astype(np.float32)
                self.dense_data[col] = np.ascontiguousarray(matrix)
            else:
                # Int array -> Int64 (Long)
                self.scalar_data[col] = series.to_numpy(dtype=np.int64)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Zero-copy conversion to tensors
        batch = {}
        for col, data in self.dense_data.items():
            batch[col] = torch.from_numpy(data[idx])
        for col, data in self.scalar_data.items():
            batch[col] = torch.tensor(data[idx], dtype=torch.long)
        return batch
