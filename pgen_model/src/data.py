import logging
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .data_utils import data_import_normalize, drugs_to_atc, UNKNOWN_TOKEN

logger = logging.getLogger(__name__)


class PGenDataProcess:
    def __init__(self):
        self.encoders = {}
        self.feature_cols = []
        self.target_cols = []
        self.multi_label_cols = set()
        self.cols_to_process = []

    def _split_labels(self, label_str: str) -> List[str]:
        if not isinstance(label_str, str) or not label_str:
            return []
        return [s.strip() for s in re.split(r"[|;,]", label_str) if s.strip()]

    def load_data(
        self,
        csv_path: Union[str, Path],
        all_cols: List[str],
        cols_to_use: List[str],
        input_cols: List[str],
        target_cols: List[str],
        multi_label_targets: Optional[List[str]],
        stratify_cols: Union[List[str], str],
    ) -> pd.DataFrame:
        csv_path = Path(csv_path)
        logger.info(f"Cargando datos desde {csv_path}...")

        # Carga eficiente: leer solo lo necesario si es posible
        try:
            df = pd.read_csv(
                csv_path, sep="\t", usecols=lambda c: c in cols_to_use, dtype=str
            )
        except ValueError:
            df = pd.read_csv(csv_path, sep="\t", dtype=str)

        # Normalizar
        df = data_import_normalize(
            df, all_cols, target_cols, multi_label_targets, stratify_cols
        )

        # Generar ATC si es necesario
        if "drug" in df.columns:
            df = drugs_to_atc(df, drug_col="drug", atc_col="atc")

        # Configurar columnas (minúsculas forzadas por data_import_normalize)
        self.feature_cols = [c.lower() for c in input_cols]
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = set(t.lower() for t in (multi_label_targets or []))

        # Añadir ATC a features si se generó y se requiere
        if "atc" in df.columns and "atc" not in self.feature_cols:
            # Solo si el modelo espera 'atc', lo añadimos.
            # Aquí asumimos que si se generó, es útil.
            # Ajustar lógica según configuración estricta si es necesario.
            pass

        self.cols_to_process = list(set(self.feature_cols + self.target_cols))

        # Filtrar clases minoritarias en estratificación para evitar errores en split
        counts = df["stratify_col"].value_counts()
        valid_strata = counts[counts > 1].index
        df = df[df["stratify_col"].isin(valid_strata)].reset_index(drop=True)

        # Pre-split de multilabel strings a listas
        for col in self.multi_label_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._split_labels)

        return df

    def fit(self, df_train: pd.DataFrame) -> None:
        """Ajusta encoders solo con train set."""
        logger.info("Ajustando encoders...")
        missing = set(self.cols_to_process) - set(df_train.columns)
        if missing:
            logger.warning(f"Columnas faltantes en fit: {missing}")
            self.cols_to_process = [
                c for c in self.cols_to_process if c in df_train.columns
            ]

        for col in self.cols_to_process:
            if col in self.multi_label_cols:
                enc = MultiLabelBinarizer()
                enc.fit(df_train[col])
                self.encoders[col] = enc
            else:
                enc = LabelEncoder()
                # Añadir UNKNOWN_TOKEN explícitamente
                uniques = df_train[col].astype(str).unique().tolist()
                if UNKNOWN_TOKEN not in uniques:
                    uniques.append(UNKNOWN_TOKEN)
                enc.fit(sorted(uniques))
                self.encoders[col] = enc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma usando encoders ajustados."""
        if not self.encoders:
            raise RuntimeError("Llamar a fit() antes de transform()")

        df_out = df.copy()
        for col, enc in self.encoders.items():
            if col not in df_out.columns:
                continue

            if isinstance(enc, MultiLabelBinarizer):
                # Asegurar listas (aunque _split_labels ya debería haberlo hecho)
                # Esto es una salvaguarda si la columna no fue pre-procesada.
                df_out[col] = df_out[col].apply(
                    lambda x: x if isinstance(x, list) else []
                )
                # MLB devuelve matriz numpy, la guardamos como una serie de arrays numpy
                encoded_data = enc.transform(df_out[col])
                df_out[col] = pd.Series(list(encoded_data), index=df_out.index)

            elif isinstance(enc, LabelEncoder):
                # Manejo robusto de desconocidos vectorizado
                vals = df_out[col].astype(str).to_numpy()

                # Usar búsqueda rápida con np.isin
                mask = ~np.isin(vals, enc.classes_)
                if mask.any():
                    vals[mask] = UNKNOWN_TOKEN

                df_out[col] = enc.transform(vals)

        return df_out


class PGenDataset(Dataset):
    """
    Dataset optimizado para memoria.
    Mantiene los datos en numpy arrays y solo convierte a Tensor al servirlos.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        multi_label_cols: set,
    ):
        self.data = {}
        self.feature_cols = [f.lower() for f in feature_cols]
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = multi_label_cols

        # Validar columnas
        cols_needed = [
            c for c in (self.feature_cols + self.target_cols) if c in df.columns
        ]

        for col in cols_needed:
            series = df[col]
            if col in self.multi_label_cols:
                # Convertir lista de listas/arrays a un único ndarray 2D
                try:
                    # Asumiendo que transform() dejó listas o arrays en las celdas
                    # np.vstack es eficiente si la entrada es lista de arrays
                    matrix = np.vstack(series.values).astype(np.float32)  # type: ignore
                    self.data[col] = matrix
                except Exception as e:
                    logger.error(f"Error apilando columna multilabel {col}: {e}")
                    raise
            else:
                # Single label -> array de enteros
                # Usar tipos de datos más pequeños si es posible para ahorrar memoria
                # int32 suele ser suficiente para índices de clases
                self.data[col] = series.values.astype(np.int32)

        self.length = len(df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Conversión on-the-fly a Tensor.
        # Esto es muy rápido y evita tener todo duplicado en memoria Tensor.
        batch = {}
        for col, array in self.data.items():
            # torch.from_numpy comparte memoria si es posible.
            # Al hacer slicing [idx], numpy devuelve una copia si es un slice complejo,
            # pero para un escalar (single label) devuelve un valor.
            val = array[idx]
            if isinstance(val, np.ndarray):
                # Multi-label: es un array, convertir a tensor float
                tensor = torch.from_numpy(val)
            else:
                # Single-label: es un escalar (numpy scalar), convertir a tensor long
                tensor = torch.tensor(val, dtype=torch.long)

            batch[col] = tensor
        return batch
