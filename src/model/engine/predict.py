# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning
# Copyright (C) 2025  Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
predict.py

Inference Engine for Pharmagen.
Reconstructs the model from training artifacts (config snapshots + encoders)
and performs optimized inference (single-sample, DataFrame batch, or CSV file).
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch

from src.cfg.manager import DIRS, MULTI_LABEL_COLS, get_model_config
from src.interface.ui import ProgressBar
from src.model.architecture.deep_fm import ModelConfig, PharmagenDeepFM

logger = logging.getLogger(__name__)
MIN_SIZE_PBAR = 2000  # Minimum samples to show progress bar


class PGenPredictor:
    """
    Stateful Inference Engine.

    Loads trained artifacts (Encoders, Weights, Config) into memory
    to provide a unified interface for prediction.

    Args:
        model_name (str): Identifier of the model to load.
        device (str, optional): Computation device ('cpu', 'cuda').
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.unknown_token = "__UNKNOWN__"

        logger.info("Initializing Predictor for %s on %s...", self.model_name, self.device)

        # 1. Load Artifacts
        self.config_snapshot = self._load_config_snapshot()
        self.encoders = self._load_encoders()

        # 2. Reconstruct Architecture
        self.model = self._reconstruct_model()
        self.model.eval()

        # 3. Optimization: Cache Unknown Indices
        self.unknown_indices = self._cache_unknown_indices()

        # Public metadata
        self.feature_cols = list(self.model.cfg.n_features.keys())
        self.target_cols = list(self.model.cfg.target_dims.keys())

    def _load_config_snapshot(self) -> dict[str, Any]:
        """Loads the JSON config snapshot saved at the end of training."""
        snapshot_path = DIRS["reports"] / f"{self.model_name}_final_config.json"

        if snapshot_path.exists():
            logger.debug("Loading config snapshot from %s", snapshot_path)
            with open(snapshot_path) as f:
                return json.load(f)
        else:
            logger.warning("Snapshot not found at %s. Falling back to defaults.", snapshot_path)
            # Partial fallback logic
            return {"model_config": {}}

    def _load_encoders(self) -> dict[str, Any]:
        """Loads the pickled sklearn encoders."""
        path = DIRS["encoders"] / f"encoders_{self.model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Encoders not found at {path}. Model must be trained first.")
        return joblib.load(path)

    def _reconstruct_model(self) -> PharmagenDeepFM:
        """
        Recreates the PyTorch model structure and loads weights.
        """
        # 1. Dimensions Extraction (Source of Truth: Encoders)
        n_features: dict[str, int] = {}
        target_dims: dict[str, int] = {}

        # Fallback to Config for column lists if snapshot missing
        if "data_config" in self.config_snapshot:
            f_cols = self.config_snapshot["data_config"].get("feature_cols", [])
            t_cols = self.config_snapshot["data_config"].get("target_cols", [])
        else:
            raw_cfg = get_model_config(self.model_name)
            f_cols = raw_cfg.get("data", {}).get("features", [])
            t_cols = raw_cfg.get("data", {}).get("targets", [])

        # Map dimensions
        for col, enc in self.encoders.items():
            if hasattr(enc, "classes_"):
                dim = len(enc.classes_)
                if col in f_cols:
                    n_features[col] = dim
                elif col in t_cols:
                    target_dims[col] = dim

        # 2. Build Config Object
        full_cfg = self.config_snapshot.get("full_config", {})
        arch_params = full_cfg.get("architecture", {})

        # If empty (legacy snapshot), try 'model_config'
        if not arch_params:
            arch_params = self.config_snapshot.get("model_config", {})

        # Construct safe config
        config = ModelConfig(
            n_features=n_features,
            target_dims=target_dims,
            embedding_dim=arch_params.get("embedding_dim", 64),
            embedding_dropout=arch_params.get("embedding_dropout", 0.1),
            hidden_dim=arch_params.get("hidden_dim", 256),
            n_layers=arch_params.get("n_layers", 3),
            dropout_rate=arch_params.get("dropout_rate", 0.2),
            activation=arch_params.get("activation", "gelu"),
            use_transformer=arch_params.get("use_transformer", True),
            attn_dim_feedforward=arch_params.get("attn_dim_feedforward", 512),
            attn_heads=arch_params.get("attn_heads", 4),
            num_attn_layers=arch_params.get("num_attn_layers", 2),
            fm_hidden_dim=arch_params.get("fm_hidden_dim", 64),
            fm_hidden_layers=arch_params.get("fm_hidden_layers", 1),
        )

        # 3. Instantiate & Load Weights
        model = PharmagenDeepFM(config)

        # Determine weights path
        weights_path = DIRS["models"] / "model_best.pt"  # Default name

        if not weights_path.exists():
            weights_path = DIRS["models"] / f"pmodel_{self.model_name}.pt"

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found at {weights_path}")

        logger.debug("Loading weights from %s", weights_path)
        state = torch.load(weights_path, map_location=self.device)

        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

        return model.to(self.device)

    def _cache_unknown_indices(self) -> dict[str, int]:
        """Pre-computes index of UNKNOWN token for fast lookup."""
        indices = {}
        for col, enc in self.encoders.items():
            if isinstance(enc, LabelEncoder):
                if self.unknown_token in enc.classes_:
                    indices[col] = int(enc.transform([self.unknown_token])[0])  # type: ignore
                else:
                    indices[col] = 0
        return indices

    # ==========================================================================
    # PUBLIC API
    # ==========================================================================

    def predict_single(self, input_dict: dict[str, Any]) -> dict[str, Any] | None:
        """
        Predicts a single dictionary sample.

        Args:
            input_dict: {'drug': 'Aspirin', 'gene': 'CYP2D6'}

        Returns:
            dict: Decoded predictions or None on failure.
        """
        model_inputs = {}
        try:
            # Pre-normalize input keys for efficient lookup (O(1) instead of O(n) per column)
            input_dict_lower = {k.lower(): v for k, v in input_dict.items()}
            
            for col in self.feature_cols:
                # Direct lookup with pre-normalized keys
                val = input_dict_lower.get(col)
                
                if val is None:
                    raise ValueError(f"Missing input feature: {col}")

                # Prepare tensor (1,)
                model_inputs[col] = self._prepare_tensor(col, val)

            with torch.inference_mode():
                outputs = self.model(model_inputs)

            return self._decode(outputs)[0]

        except Exception as e:
            logger.error("Prediction failed: %s", e)
            return None

    def predict_dataframe(self, df: pd.DataFrame, batch_size: int = 512) -> list[dict[str, Any]]:
        """
        Predicts efficiently on a Pandas DataFrame in chunks.

        Handles:
        - Column case normalization
        - Efficient batching (Tensor movement to GPU inside loop)
        - Memory safety for large DataFrames

        Args:
            df (pd.DataFrame): Input data.
            batch_size (int): Inference batch size.

        Returns:
            list[dict]: List of prediction dictionaries.
        """
        # 1. Normalize columns (in-place for efficiency)
        df.columns = df.columns.str.lower().str.strip()

        # 2. Validate Schema
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            logger.error("DataFrame missing required columns: %s", missing)
            return []

        # 3. Pre-process to CPU Tensors (Vectorized)
        tensor_dict = {}
        for col in self.feature_cols:
            tensor_dict[col] = self._prepare_batch_tensor(col, df[col])

        # 4. Batch Loop
        num_samples = len(df)
        all_results = []

        # Iteration Strategy
        iterator = range(0, num_samples, batch_size)
        use_pbar = num_samples > MIN_SIZE_PBAR

        with torch.inference_mode():
            if use_pbar:
                # Custom ProgressBar
                pbar = ProgressBar(total=num_samples, desc="Inference", width=30)
                pbar.__enter__()

            try:
                for i in iterator:
                    end = min(i + batch_size, num_samples)
                    current_batch_size = end - i

                    # Create Batch & Move to Device
                    batch_inputs = {
                        k: v[i:end].to(self.device, non_blocking=True)
                        for k, v in tensor_dict.items()
                    }

                    # Forward
                    outputs = self.model(batch_inputs)

                    # Decode (CPU side)
                    decoded_batch = self._decode(outputs)
                    all_results.extend(decoded_batch)

                    if use_pbar:
                        pbar.update(current_batch_size)  # type: ignore
            finally:
                if use_pbar:
                    pbar.__exit__(None, None, None)  # type: ignore

        return all_results

    def predict_file(self, file_path: str | Path, batch_size: int = 512) -> list[dict[str, Any]]:
        """
        Wrapper for batch prediction from a file path.
        Supports CSV and TSV.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("File not found: %s", path)
            return []

        try:
            logger.info("Reading %s...", path.name)
            # Detect separator
            sep = "\t" if path.suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)

            return self.predict_dataframe(df, batch_size)

        except Exception as e:
            logger.error("Failed to process file %s: %s", path, e)
            return []

    # ==========================================================================
    # INTERNAL HELPERS
    # ==========================================================================

    def _prepare_tensor(self, col: str, val: Any) -> torch.Tensor:
        """Encodes a scalar value to tensor (On Device)."""
        enc = self.encoders[col]
        val_str = str(val)

        if val_str in enc.classes_:
            idx = int(enc.transform([val_str])[0])
        else:
            idx = self.unknown_indices.get(col, 0)

        return torch.tensor([idx], dtype=torch.long, device=self.device)

    def _prepare_batch_tensor(self, col: str, series: pd.Series) -> torch.Tensor:
        """Encodes a Series to a Tensor (On CPU)."""
        enc = self.encoders[col]
        # Robust string conversion handling NaNs
        vals = series.fillna(self.unknown_token).astype(str).to_numpy()

        # Identify knowns
        mask = np.isin(vals, enc.classes_)

        # Replace unknowns
        vals[~mask] = self.unknown_token

        # Safety fallback if unknown token itself is missing in encoder
        if self.unknown_token not in enc.classes_:
            encoded = np.zeros(len(vals), dtype=int)  # All to 0
            known_vals = vals[mask]
            if len(known_vals) > 0:
                encoded[mask] = enc.transform(known_vals)
        else:
            encoded = enc.transform(vals)

        # Return CPU tensor
        return torch.from_numpy(encoded).long()

    def _decode(self, outputs: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
        """Decodes raw logits to readable labels."""
        first = next(iter(outputs.values()))
        bs = first.size(0)
        decoded_cols = {}

        for col, logits in outputs.items():
            enc = self.encoders[col]
            logits.cpu()

            if col in MULTI_LABEL_COLS:
                # Multi-label
                probs = torch.sigmoid(logits)
                preds = (probs > (1 / 2)).numpy()
                labels = enc.inverse_transform(preds)
                decoded_cols[col] = [list(x) for x in labels]
            else:
                # Multi-class
                preds = torch.argmax(logits, dim=1).numpy()
                labels = enc.inverse_transform(preds)
                decoded_cols[col] = labels.tolist()

        # Transpose
        results = []
        for i in range(bs):
            row = {k: decoded_cols[k][i] for k in decoded_cols}
            results.append(row)

        return results
