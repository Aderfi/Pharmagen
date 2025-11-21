# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import re
from pathlib import Path
from typing import List, Optional, Union, Set, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.data import normalize_dataframe, drugs_to_atc, UNKNOWN_TOKEN

logger = logging.getLogger(__name__)


class PGenDataProcess(BaseEstimator, TransformerMixin):
    def __init__(
            self, 
            feature_cols: List[str], 
            target_cols: List[str], 
            multi_label_cols: Optional[List[str]] = None
        ):
        """
        Inicializa el procesador con la configuración FINAL de columnas.
        Ya no depende de cargar un CSV para saber qué columnas procesar.
        """
        self.feature_cols = [c.lower() for c in feature_cols]
        self.target_cols = [c.lower() for c in target_cols]
        self.multi_label_cols = set(c.lower() for c in (multi_label_cols or []))
        
        # Calculamos qué columnas procesar desde el inicio
        self.cols_to_process = list(set(self.feature_cols + self.target_cols))
        self.encoders: Dict = {}

    @staticmethod
    def _split_labels(label_str: Union[str, float]) -> List[str]:
        if not isinstance(label_str, str) or not label_str:
            return []
        return [s.strip() for s in re.split(r"[|;,]", label_str) if s.strip()]

    def fit(self, df_train: pd.DataFrame, y=None):
        """Ajusta encoders solo con train set.
        
        Args:
            df_train (pd.DataFrame): DataFrame con datos de entrenamiento.
            y (pd.Series, optional): Series con datos de entrenamiento. Por defecto None.
        
        Returns:
            self: Retorna el objeto ajustado.
        
        >>> pgen_data_process = PGenDataProcess(
        ...     feature_cols=["feature1", "feature2"],
        ...     target_cols=["target1", "target2"],
        ...     multi_label_cols=["target1", "target2"]
        ... )
        >>> pgen_data_process.fit(df_train)
        """
        logger.info("Ajustando encoders...")
        
        # Validación rápida
        missing = set(self.cols_to_process) - set(df_train.columns)
        if missing:
            logger.warning(f"Columnas esperadas no encontradas en fit: {missing}")
            # Ajustamos dinámicamente solo a lo que existe (para robustez)
            self.cols_to_process = [c for c in self.cols_to_process if c in df_train.columns]

        for col in self.cols_to_process:
            # Pre-procesamiento 'on-the-fly' para el ajuste si viene sucio
            series = df_train[col]
            
            if col in self.multi_label_cols:
                # Aseguramos que sea lista antes de ajustar
                if not series.empty and isinstance(series.iloc[0], str):
                     series = series.apply(self._split_labels)
                
                enc = MultiLabelBinarizer()
                enc.fit(series)
                self.encoders[col] = enc
            else:
                enc = LabelEncoder()
                # Convertir a string y manejar nulos como string vacío o token
                uniques = series.fillna("").astype(str).unique().tolist()
                if UNKNOWN_TOKEN not in uniques:
                    uniques.append(UNKNOWN_TOKEN)
                enc.fit(sorted(uniques))
                self.encoders[col] = enc
        
        return self # Permite encadenamiento (method chaining)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma el DataFrame aplicando los encoders ajustados.
        
        Args:
            df (pd.DataFrame): DataFrame con datos a transformar.
        
        Returns:
            pd.DataFrame: DataFrame transformado con encoders aplicados.
        
        Raises:
            RuntimeError: Si el procesador no ha sido ajustado (fit() no llamado).
        """
        if not self.encoders:
            raise RuntimeError("El procesador no ha sido ajustado. Llama a fit() primero.")

        df_out = df.copy()
        
        for col, enc in self.encoders.items():
            if col not in df_out.columns:
                continue

            # 1. Manejo de MultiLabel
            if isinstance(enc, MultiLabelBinarizer):
                # Normalización de entrada: String -> Lista
                if not df_out[col].empty and isinstance(df_out[col].iloc[0], str):
                     df_out[col] = df_out[col].apply(self._split_labels)
                
                # Salvaguarda para nulos/no-listas
                df_out[col] = df_out[col].apply(lambda x: x if isinstance(x, list) else [])
                
                encoded_data = enc.transform(df_out[col])
                # Guardamos como objeto (array numpy dentro de la celda) para mantener estructura DataFrame
                df_out[col] = pd.Series(list(encoded_data), index=df_out.index)

            # 2. Manejo de LabelEncoder (Variables simples)
            elif isinstance(enc, LabelEncoder):
                vals = df_out[col].fillna("").astype(str).to_numpy()
                
                # Manejo vectorizado de desconocidos
                mask = ~np.isin(vals, enc.classes_)
                if mask.any():
                    vals[mask] = UNKNOWN_TOKEN
                
                df_out[col] = enc.transform(vals)

        return df_out

class PGenDataset(Dataset):
    """
    PyTorch Dataset Class.

    Separa internamente escalares y matrices para eliminar condicionales en el ciclo de entrenamiento.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
        multi_label_cols: Set[str],
    ):
        # 1. Normalización de columnas
        self.feature_cols = [f.lower() for f in feature_cols]
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = {c.lower() for c in multi_label_cols}
        
        self._arrays_data: Dict[str, np.ndarray] = {} # multilabel/embeddings (matrices 2D)
        self._scalars_data: Dict[str, np.ndarray] = {} # single label/índices (arrays 1D)

        cols_to_process = [
            c for c in (self.feature_cols + self.target_cols) if c in df.columns
        ]

        # 2. Carga y conversión masiva
        for col in cols_to_process:
            series = df[col]
            
            if col in self.multi_label_cols:
                self._process_multi_label(col, series)
            else:
                self._process_single_label(col, series)

        self.length = len(df)

    def _process_multi_label(self, col: str, series: pd.Series):
        """Maneja columnas que contienen listas/arrays (MultiLabel)."""
        try:
            # TRUCO DE RENDIMIENTO:
            # series.values en columnas 'object' devuelve un array de referencias.
            # series.tolist() crea una lista de python pura.
            # np.stack(...) sobre una lista es mucho más rápido que sobre un array de objetos.
            matrix = np.stack(series.tolist()).astype(np.float32)
            
            # Asegurar memoria contigua para que torch.from_numpy no haga copias
            self._arrays_data[col] = np.ascontiguousarray(matrix)
            
        except ValueError as e:
            logger.error(f"Error de dimensionalidad en columna '{col}'. Verifica que todas las filas tengan la misma longitud.")
            raise e
        except Exception as e:
            logger.error(f"Error procesando multilabel '{col}': {e}")
            raise

    def _process_single_label(self, col: str, series: pd.Series):
        """Maneja columnas de valores simples (Single Label)."""
        # Para single labels (índices de clase), PyTorch usa Long (int64).
        # Si la memoria es crítica, usar int32.
        self._scalars_data[col] = series.values.astype(np.int64)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 3. Acceso Optimizado
        batch = {}

        # Procesar Matrices (Multi-label / Features densas) -> Float Tensor
        for col, data in self._arrays_data.items():
            # torch.from_numpy crea un tensor compartiendo memoria (zero-copy)
            # data[idx] devuelve un slice (view) o copia según memoria. 
            batch[col] = torch.from_numpy(data[idx])

        # Procesar Escalares (Single-label) -> Long Tensor
        for col, data in self._scalars_data.items():
            # data[idx] devuelve un numpy scalar (ej: numpy.int64)
            # from_numpy no acepta escalares.
            batch[col] = torch.tensor(data[idx], dtype=torch.long)

        return batch
