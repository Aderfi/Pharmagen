import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from torch.utils.data import Dataset

# Configuración de Logging
logger = logging.getLogger("Pharmagen.Data")

# =============================================================================
# 1. CONFIGURATION LAYER
# =============================================================================

@dataclass(frozen=True)
class DataConfig:
    """
    Centralized configuration for Data Loading and Processing.
    """
    # File Paths
    dataset_path: Path
    separator: str = "\t"

    # Schema Definition
    feature_cols: list[str] = field(default_factory=list)
    target_cols: list[str] = field(default_factory=list)
    multi_label_cols: list[str] = field(default_factory=list)

    # Cleaning & Stratification
    stratify_col: str | None = None
    min_category_count: int = 1  # Threshold to handle rare categories
    unknown_token: str = "__UNKNOWN__"

    # Technical
    num_workers: int = 4
    pin_memory: bool = True

    @property
    def all_cols(self) -> list[str]:
        return self.feature_cols + self.target_cols

# =============================================================================
# 2. ROBUST DATA LOADING
# =============================================================================

def load_and_clean_dataset(config: DataConfig) -> pd.DataFrame:
    """
    Loads dataset efficiently, validating schema and performing initial cleanup.
    """
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {config.dataset_path}")

    logger.info(f"Loading dataset from {config.dataset_path.name}...")

    # 1. Load only necessary columns to save memory
    try:
        df = pd.read_csv(
            config.dataset_path,
            sep=config.separator,
            usecols=lambda c: c.lower().strip() in [x.lower() for x in config.all_cols] # type: ignore
        )
    except ValueError as e:
        logger.error(f"Column mismatch error. Check config matches CSV headers. Details: {e}")
        raise

    # Normalize headers
    df.columns = df.columns.str.lower().str.strip()

    # 2. Vectorized String Cleaning (Avoid loops where possible)
    # Compiling regex once for performance
    delimiters_re = re.compile(r"[,;]+")

    multi_label_set = set(c.lower() for c in config.multi_label_cols)

    for col in df.columns:
        # Fast conversion to string and lowercasing
        series = df[col].fillna(config.unknown_token).astype(str).str.strip().str.lower()

        if col in multi_label_set:
            # Normalize delimiters to pipe '|' for multi-label consistency
            df[col] = series.apply(lambda x: delimiters_re.sub("|", x))
        else:
            df[col] = series

    # 3. Stratification Logic (Safe)
    if config.stratify_col:
        s_cols = [c.strip().lower() for c in config.stratify_col.split(",")]
        valid_s_cols = [c for c in s_cols if c in df.columns]

        if valid_s_cols:
            # Create a stratification key
            strat_key = df[valid_s_cols].astype(str).agg("|".join, axis=1)

            # Filter singletons efficiently
            counts = strat_key.value_counts()
            valid_keys = counts[counts > 1].index

            initial_len = len(df)
            df = df[strat_key.isin(valid_keys)].reset_index(drop=True)
            logger.info(f"Stratification dropped {initial_len - len(df)} singleton rows.")
        else:
            logger.warning("Stratification columns requested but not found in dataframe.")

    return df

# =============================================================================
# 3. PREPROCESSING ENGINE
# =============================================================================

class PGenProcessor(BaseEstimator, TransformerMixin):
    """
    Stateful processor that converts DataFrame -> Dictionary of Tensors/Arrays.
    Decouples Pandas overhead from the training loop.
    """
    def __init__(self, config: DataConfig):
        self.cfg = config
        self.encoders: dict[str, Any] = {}
        # Pre-compute sets for O(1) lookups
        self.multi_label_set = set(c.lower() for c in config.multi_label_cols)
        self.scalar_cols = [c.lower() for c in config.feature_cols if c.lower() not in self.multi_label_set]
        self.target_cols = [c.lower() for c in config.target_cols]

    def fit(self, df: pd.DataFrame, y=None):
        logger.info("Fitting feature encoders...")

        # Fit Scalars (LabelEncoding)
        for col in self.scalar_cols + self.target_cols:
            if col not in df.columns:
                continue

            # Handle unknown tokens logic
            uniques = set(df[col].unique())
            uniques.add(self.cfg.unknown_token)

            enc = LabelEncoder()
            enc.fit(sorted(list(uniques)))
            self.encoders[col] = enc

        # Fit Multi-Labels (Binarizer)
        for col in self.multi_label_set:
            if col not in df.columns:
                continue

            # Efficient splitting
            parsed = df[col].str.split("|").map(lambda x: x if isinstance(x, list) else [])
            enc = MultiLabelBinarizer()
            enc.fit(parsed)
            self.encoders[col] = enc

        return self

    def transform(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """
        Transforms DataFrame directly into a Dictionary of Tensors ready for PGenDataset.
        This avoids storing arrays inside Pandas Series.
        """
        output_payload = {}

        # 1. Transform Scalars
        for col in self.scalar_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]

            # Safe transform handling unseen labels
            vals = df[col].to_numpy()
            # Fast check using numpy set operations is complex, relying on map/apply is safer for strings
            # Optimization: Use searchsorted if we converted to int codes earlier, but strings strictly need mapping.
            # We assume fit included UNKNOWN. We map unseen to UNKNOWN index.

            # Robust Transform
            unknown_idx = np.where(enc.classes_ == self.cfg.unknown_token)[0][0] # noqa

            # Check validity
            valid_mask = np.isin(vals, enc.classes_)
            vals[~valid_mask] = self.cfg.unknown_token

            encoded_data = enc.transform(vals)
            output_payload[col] = torch.tensor(encoded_data, dtype=torch.long)

        # 2. Transform Targets
        for col in self.target_cols:
            if col not in df.columns:
                continue
            # Targets usually don't have "Unknowns" in training, but safe to handle same way
            enc = self.encoders[col]
            encoded_data = enc.transform(df[col].to_numpy())
            output_payload[col] = torch.tensor(encoded_data, dtype=torch.long) # Or Float if regression

        # 3. Transform Multi-Labels
        for col in self.multi_label_set:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            parsed = df[col].str.split("|").map(lambda x: x if isinstance(x, list) else [])

            # transform returns a dense numpy array (N, Num_Classes)
            mat = enc.transform(parsed).astype(np.float32)
            output_payload[col] = torch.from_numpy(mat)

        output_payload = cast(dict[str, torch.Tensor], output_payload)
        return output_payload

# =============================================================================
# 4. MEMORY-OPTIMIZED DATASET
# =============================================================================

class PGenDataset(Dataset):
    """
    Zero-copy Dataset.
    It expects pre-tensed data. It does NOT do any processing in __getitem__.
    """
    def __init__(self, data_payload: dict[str, torch.Tensor]):
        """
        Args:
            data_payload: Dictionary containing full-batch tensors.
                          { 'gene_id': Tensor(N,), 'fingerprints': Tensor(N, 1024), ... }
        """
        self.data = data_payload
        # Verify alignment
        lengths = [len(t) for t in self.data.values()]
        if not all(l == lengths[0] for l in lengths): # noqa
            raise ValueError(f"Tensor length mismatch in dataset: {lengths}")
        self.length = lengths[0]
        self.keys = list(self.data.keys())

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Extremely fast lookup.
        # Slicing a tensor is a view, usually very cheap.
        return {key: self.data[key][idx] for key in self.keys}
