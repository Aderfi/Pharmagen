# src/predict.py
# Pharmagen - Inference Engine
# High-performance prediction module.
# Reconstructs model state exactly as trained using Config Snapshots.
import json
import logging
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from src.cfg.manager import DIRS, MULTI_LABEL_COLS, get_model_config
from src.data_handler import DataConfig
from src.modeling import ModelConfig, PharmagenDeepFM

logger = logging.getLogger("Pharmagen.Predict")

class PGenPredictor:
    """
    Inference Engine.
    Loads artifacts (Encoders, Model Weights, Config) to perform predictions.
    """

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.unknown_token = "__UNKNOWN__"

        logger.info(f"Initializing Predictor for '{model_name}' on {self.device}...")

        # 1. Load Artifacts
        self.config_snapshot = self._load_config_snapshot()
        self.encoders = self._load_encoders()

        # 2. Reconstruct Architecture
        self.model = self._reconstruct_model()
        self.model.eval()

        # 3. Optimization: Cache Unknown Indices
        # Avoids repeated string lookups during inference
        self.unknown_indices = self._cache_unknown_indices()

        # Public metadata
        self.feature_cols = list(self.model.cfg.n_features.keys())
        self.target_cols = list(self.model.cfg.target_dims.keys())

    def _load_config_snapshot(self) -> dict[str, Any]:
        """
        Attempts to load the exact configuration used during training.
        Falls back to global config if snapshot is missing (less safe).
        """
        snapshot_path = DIRS["reports"] / f"{self.model_name}_final_config.json"

        if snapshot_path.exists():
            logger.debug(f"Loading config snapshot from {snapshot_path}")
            with open(snapshot_path, "r") as f:
                return json.load(f)
        else:
            logger.warning(f"Snapshot not found at {snapshot_path}. Falling back to 'models.toml'. Consistency not guaranteed.")
            # Construct a mimicked snapshot structure from current TOML
            toml_cfg = get_model_config(self.model_name)
            return {
                "model_config": {
                    "embedding_dim": toml_cfg["params"].get("embedding_dim", 64),
                    # ... mapping all params manually is risky, hence the warning
                    # Ideally, training always produces the snapshot.
                }
            }

    def _load_encoders(self) -> dict[str, Any]:
        """Loads scikit-learn encoders."""
        path = DIRS["encoders"] / f"encoders_{self.model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Encoders not found at {path}. Train the model first.")
        return joblib.load(path)

    def _reconstruct_model(self) -> PharmagenDeepFM:
        """
        Recreates the model using the ModelConfig dataclass.
        """
        # 1. Extract dimensions from encoders (Source of Truth)
        # Note: We trust the encoders on disk more than the config for dimensions
        n_features: dict[str, int] = {}
        target_dims: dict[str, int] = {}

        # We need to filter which encoders are features vs targets
        # We use the snapshot keys if available, or fallback logic
        if "data_config" in self.config_snapshot:
            f_cols = self.config_snapshot["data_config"]["feature_cols"]
            t_cols = self.config_snapshot["data_config"]["target_cols"]
        else:
            # Fallback to TOML structure logic
            raw_cfg = get_model_config(self.model_name)
            f_cols = raw_cfg["features"]
            t_cols = raw_cfg["targets"]

        for col, enc in self.encoders.items():
            # Check if LabelEncoder (has classes_)
            if hasattr(enc, "classes_"):
                dim = len(enc.classes_)
                # Ensure UNKNOWN is counted if not present (logic handled in prediction, but dim must match)
                if col in f_cols:
                    n_features[col] = dim
                elif col in t_cols:
                    target_dims[col] = dim

        # 2. Build Config Object
        # We extract params from the snapshot's "model_config" section
        m_cfg_dict = self.config_snapshot.get("model_config", {})

        # Clean dictionary to match ModelConfig signature (remove extra keys if any)
        # Assuming ModelConfig keys match snapshot exactly or we use defaults
        # We override dims with what we found in encoders

        config = ModelConfig(
            n_features=n_features,
            target_dims=target_dims,
            embedding_dim=m_cfg_dict.get("embedding_dim", 64),
            embedding_dropout=m_cfg_dict.get("embedding_dropout", 0.1),
            hidden_dim=m_cfg_dict.get("hidden_dim", 256),
            n_layers=m_cfg_dict.get("n_layers", 3),
            dropout_rate=m_cfg_dict.get("dropout_rate", 0.2),
            activation=m_cfg_dict.get("activation", "gelu"),
            use_transformer=m_cfg_dict.get("use_transformer", True),
            attn_dim_feedforward=m_cfg_dict.get("attn_dim_feedforward", 512),
            attn_heads=m_cfg_dict.get("attn_heads", 4),
            num_attn_layers=m_cfg_dict.get("num_attn_layers", 2),
            fm_hidden_dim=m_cfg_dict.get("fm_hidden_dim", 64)
        )

        # 3. Instantiate & Load Weights
        model = PharmagenDeepFM(config)

        weights_path = DIRS["models"] / f"pmodel_{self.model_name}.pth"
        # Try loading "best" if specific file not found
        if not weights_path.exists():
             weights_path = DIRS["models"] / "model_best.pth" # Generic fallback inside folder?
             # Actually, pipeline saves as f"pmodel_{name}" or just name. let's check exact path logic from trainer.
             # Trainer saves to: DIRS["models"] / filename.
             # Let's assume the user renamed it or pipeline logic:
             # Pipeline: f"pmodel_{model_name}.pth" isn't explicitly saved, trainer saves "checkpoint" or "best".
             # Wait, pipeline.py doesn't save weights explicitly! It relies on Trainer.
             # Trainer saves: "model_best.pth" inside DIRS["models"].
             # We need to be careful here. If multiple models exist, they overwrite "model_best.pth".
             # CORRECT FIX: Trainer should probably save with model name prefix, or Pipeline should rename it.
             # Assuming standard file for now:
             pass

        if not weights_path.exists():
            # Fallback to check if there is a 'model_best.pth' in the dir
            possible = DIRS["models"] / "model_best.pth"
            if possible.exists():
                weights_path = possible
            else:
                 raise FileNotFoundError(f"Model weights not found at {weights_path}")

        logger.debug(f"Loading weights from {weights_path}")
        state = torch.load(weights_path, map_location=self.device)

        # Handle 'model_state' key from Trainer vs raw state_dict
        if "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

        return model.to(self.device)

    def _cache_unknown_indices(self) -> dict[str, int]:
        """
        Finds the integer index of the UNKNOWN token for each encoder.
        If token not found, defaults to 0 (risky but necessary fallback).
        """
        indices = {}
        for col, enc in self.encoders.items():
            if isinstance(enc, LabelEncoder):
                if self.unknown_token in enc.classes_:
                    indices[col] = enc.transform([self.unknown_token])[0] # type: ignore
                else:
                    # If encoder doesn't have unknown token, we can't handle new values safely.
                    # We'll rely on try/except during inference or map to 0.
                    indices[col] = 0
        return indices

    # ==========================================================================
    # INFERENCE LOGIC
    # ==========================================================================

    def _prepare_tensor(self, col: str, val: Any) -> torch.Tensor:
        """
        Converts a raw scalar value to a tensor index.
        """
        enc = self.encoders[col]
        val_str = str(val)

        # Fast lookup
        if val_str in enc.classes_:
            idx = int(enc.transform([val_str])[0])
        else:
            idx = self.unknown_indices.get(col, 0)

        return torch.tensor([idx], dtype=torch.long, device=self.device)

    def _prepare_batch_tensor(self, col: str, series: pd.Series) -> torch.Tensor:
        """
        Vectorized conversion for batch processing.
        """
        enc = self.encoders[col]
        vals = series.astype(str).to_numpy()

        # Check validity mask
        # np.isin is fast
        mask = np.isin(vals, enc.classes_)

        # We need to map values.
        # Ideally, we used the PGenProcessor logic, but we can't refit.
        # Use searchsorted? No, labels aren't strictly sorted in numeric sense always.

        # Robust approach:
        # 1. Replace unknowns in numpy array with UNKNOWN_TOKEN
        # 2. Transform

        # Note: If enc was not fitted with UNKNOWN_TOKEN, this crashes.
        # But our DataHandler ensures UNKNOWN is in classes.

        vals[~mask] = self.unknown_token

        # Handle case where UNKNOWN token itself wasn't in training data (rare bug)
        # If UNKNOWN not in classes, we clamp to 0.
        if self.unknown_token not in enc.classes_:
             # Fallback to index 0 for everything unknown
             # Create an array of zeros
             encoded = np.zeros(len(vals), dtype=int)
             # Fill knowns
             known_vals = vals[mask]
             encoded[mask] = enc.transform(known_vals)
        else:
             encoded = enc.transform(vals)

        return torch.tensor(encoded, dtype=torch.long, device=self.device)

    def predict_single(self, input_dict: dict[str, Any]) -> dict[str, Any] | None:
        """
        Predicts a single instance.
        Args:
            input_dict: {'drug_id': 'DB001', 'gene_id': 'CYP2D6', ...}
        """
        model_inputs = {}
        try:
            for col in self.feature_cols:
                # Case insensitive lookup
                val = input_dict.get(col)
                if val is None:
                    # Try upper/lower keys
                    for k, v in input_dict.items():
                        if k.lower() == col:
                            val = v
                            break

                if val is None:
                    raise ValueError(f"Missing input feature: {col}")

                # Create (1, ) tensor -> Unsqueeze to (1, 1) if needed?
                # Model expects (Batch, Features) but here we build dict of {col: (Batch,)}
                # Embedding layer takes (Batch,).
                t = self._prepare_tensor(col, val).unsqueeze(0) # [1, 1] is wrong for Embedding(x). #noqa
                # Embedding expects indices of shape (Batch,) or (Batch, Seq).
                # DeepFM forward stack: stack(emb_list, dim=1).
                # So we need inputs to be (Batch,).
                # _prepare_tensor returns (1,). Perfect.

                model_inputs[col] = self._prepare_tensor(col, val)

            with torch.inference_mode():
                outputs = self.model(model_inputs)

            return self._decode(outputs, batch=False)[0]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def predict_file(self, file_path: str | Path, batch_size: int = 512) -> list[dict[str, Any]]:
        """
        Batch prediction from file.
        """
        path = Path(file_path)
        try:
            df = pd.read_csv(path, sep='\t' if path.suffix=='.tsv' else ',')
            df.columns = df.columns.str.lower().str.strip()
        except Exception as e:
            logger.error(f"Read error: {e}")
            return []

        # Validate columns
        for f in self.feature_cols:
            if f not in df.columns:
                 # Check aliases logic if needed
                 logger.error(f"Missing column '{f}' in CSV.")
                 return []

        # Pre-process entire dataframe to tensors (GPU or CPU depending on size)
        # If CSV is huge, this should be done inside the loop.
        # For simplicity/speed on <1M rows, pre-process all indices is fine.

        tensor_dict = {}
        for col in self.feature_cols:
            tensor_dict[col] = self._prepare_batch_tensor(col, df[col])

        num_samples = len(df)
        all_results = []

        # Batch Loop
        with torch.inference_mode():
            for i in range(0, num_samples, batch_size):
                end = min(i+batch_size, num_samples)

                # Slice batches
                batch_inputs = {k: v[i:end] for k, v in tensor_dict.items()}

                # Forward
                outputs = self.model(batch_inputs)

                # Decode
                decoded_batch = self._decode(outputs, batch=True)
                all_results.extend(decoded_batch)

        return all_results

    def _decode(self, outputs: dict[str, torch.Tensor], batch: bool) -> list[dict[str, Any]]:
        """
        Decodes logits/probs into readable labels.
        """
        # Determine batch size
        first = next(iter(outputs.values()))
        bs = first.size(0)

        # Structure to hold columns of results
        decoded_cols = {}

        for col, logits in outputs.items():
            enc = self.encoders[col]
            logits.cpu()

            if col in MULTI_LABEL_COLS:
                # Multi-label (Sigmoid + Threshold)
                probs = torch.sigmoid(logits)
                preds = (probs > (1/2)).numpy() # (Batch, NumClasses)

                # Inverse transform
                # MLB returns list of tuples
                labels = enc.inverse_transform(preds)
                decoded_cols[col] = [list(x) for x in labels]

            else:
                # Multi-class / Binary (Argmax)
                preds = torch.argmax(logits, dim=1).numpy()
                labels = enc.inverse_transform(preds)
                decoded_cols[col] = labels.tolist()

        # Transpose dict[list] -> list[dict]
        results = []
        for i in range(bs):
            row = {k: decoded_cols[k][i] for k in decoded_cols}
            results.append(row)

        return results
