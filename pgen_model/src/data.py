from typing import Any, Union
import glob
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from torch.utils.data import Dataset

import src.config.config as cfg
from src.config.config import MODEL_TRAIN_DATA
from .model_configs import MULTI_LABEL_COLUMN_NAMES

logger = logging.getLogger(__name__)

# Constantes de configuración
MIN_SAMPLES_FOR_CLASS_GROUPING = 20  # Umbral mínimo de muestras por clase antes de agrupar

def serialize_multi_labelcols(label_input):
    """
    Convierte una lista/array en un string ordenado.
    Ejemplo: ['Nausea', 'Headache'] -> 'Headache|Nausea'
    """
    labels_list = []
    if isinstance(label_input, list):
        labels_list = [str(x).strip() for x in label_input if x is not None]
    elif isinstance(label_input, str):
        if not pd.isna(label_input) and label_input.lower() not in ["nan", "", "null"]:
             labels_list = [s.strip() for s in re.split(r"[,;]+", label_input) if s.strip()]
    elif isinstance(label_input, (list, tuple, set, np.ndarray)):
        # Si ya viene como lista (por un proceso previo)
        labels_list = [str(x).strip() for x in label_input if x is not None]
    else:
        labels_list = []

    if not labels_list:
        return "" # Retornamos string vacío para listas vacías
    
    # Ordenamos alfabéticamente para que el orden original no importe
    labels_list.sort() 
    
    # Unimos con un separador seguro (pipe |)
    return ",".join(labels_list)

class PGenDataProcess:
    """
    Clase refactorizada para preprocesar datos de farmacogenética.
    
    Esta clase ahora sigue el flujo de trabajo fit/transform de Scikit-learn
    para prevenir la fuga de datos (data leakage).
    
    1. Llama a `load_and_clean_data` para cargar y limpiar el CSV.
    2. Divide los datos (train/val/test) FUERA de esta clase.
    3. Llama a `fit` SÓLO con los datos de entrenamiento (train_df).
    4. Llama a `transform` para aplicar el preprocesamiento a 
       train_df, val_df, y test_df.
    """

    def __init__(self):
        self.encoders = {}
        self.feature_cols = set()
        self.target_cols = []
        self.multi_label_cols = set()
        
        self.cols_to_process = list(self.feature_cols.union(set(self.target_cols)))
    
        self.unknown_token = "__UNKNOWN__"
        

    def _split_labels(self, label_str) -> list[Any] | list[str | Any]:
        """Helper: Divide un string 'A, B' en una lista ['A', 'B'] y maneja nulos."""
        if not isinstance(label_str, str) \
            or pd.isna(label_str) \
            or label_str.lower() in ["nan", "", "null"]:
            return []
          
        return [s.strip() for s in re.split(r",|;", label_str) if s.strip()]

    def load_data(self, 
                  csv_path: str | Path, 
                  all_cols: List[str], 
                  input_cols: List[str],
                  target_cols: List[str], 
                  multi_label_targets: Optional[List[str]], 
                  stratify_cols: List[str]
    ) -> pd.DataFrame:
        
        """
        Load, clean and prepare data for splitting.
        
        Does NOT fit any encoders. This should be called before fit().
        
        Args:
            csv_path: Path to TSV file
            all_cols: List of column names to load
            target_cols: List of target column names
            multi_label_targets: List of multi-label target columns (or None)
            stratify_cols: List of columns to use for stratification
            
        Returns:
            Cleaned DataFrame ready for train/val/test split
            
        Raises:
            FileNotFoundError: If csv_path does not exist
            KeyError: If required columns are missing from CSV
        """
        self.cols_to_process = [c.lower() for c in all_cols]
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = set(t.lower() for t in multi_label_targets or [])
        
        # Validate that csv_path exists
        csv_path_obj = Path(csv_path)
        if not csv_path_obj.exists():
            raise FileNotFoundError(f"CSV/TSV file not found: {csv_path}")
        logger.info(f"Cargando datos desde: {csv_path}")
        
        try:
            df = pd.read_csv(str(csv_path_obj), sep="\t", index_col=False, usecols=self.cols_to_process, dtype=str)
        except Exception as e:
            raise KeyError(f"Error al leer el archivo CSV/TSV. Verifica las columnas requeridas: {self.cols_to_process}") from e
                
        df.columns = [col.lower() for col in df.columns]
                
        logging.info("Limpiando nombres de entidades (reemplazando espacios con '_')...")
        single_label_cols = [col for col in self.cols_to_process if col in df.columns and col not in self.multi_label_cols]
        multi_label_cols_present = [col for col in self.multi_label_cols if col in df.columns]
        
        # Vectorized operation for all single-label columns at once
        if single_label_cols:
            for col in single_label_cols:
                # Convertimos a string, manejando NaNs primero para no tener "nan" literal si no queremos
                df[col] = df[col].fillna("UNKNOWN").astype(str)
                
                # Si hay comas en una columna single, las reemplazamos por pipe (limpieza)
                if any(df[col].str.contains(r'[,;]+')):
                    df[col] = df[col].str.replace(r',\s+|;\s+', ',', regex=True)
                
                # Reemplazar espacios por guiones bajos
                df[col] = df[col].str.replace(' ', '_', regex=False).str.strip()
                
        if multi_label_cols_present:
            for col in multi_label_cols_present:
                # Aplicamos la limpieza y serialización ordenada
                df[col] = df[col].apply(serialize_multi_labelcols)
        del single_label_cols, multi_label_cols_present     
        logger.info("Nombres de entidades limpiados.")
        
        # 1. Normaliza columnas target (y de estratificación) a string
        all_target_and_stratify = set(self.target_cols + stratify_cols)
        for col in all_target_and_stratify:
            if col in df.columns:
                # Rellenar nulos si queda alguno y asegurar string
                df[col] = df[col].astype(str).str.strip()

        # 2. Crear 'stratify_col'
        # Usamos las columnas de estratificación proporcionadas
                
        df["stratify_col"] = df[stratify_cols].agg("_".join, axis=1)

        
        
        # 3. Filtrar datos insuficientes (opcional, pero buena práctica)
        counts = df["stratify_col"].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df["stratify_col"].isin(suficientes)].reset_index(drop=True)
        del counts, suficientes

        # 4. Procesar las columnas multi-etiqueta (convertir string a lista)
        for col in self.multi_label_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._split_labels)
        
        logger.info(f"Datos cargados y limpios. Total de filas: {len(df)}")
        return df

    def fit(self, df_train: pd.DataFrame) -> None:
        """
        Ajusta los codificadores usando SOLAMENTE datos de entrenamiento para prevenir
        fugas de datos (data leakage).

        Debe ser llamado antes de transform().

        Args:
            df_train: DataFrame de entrenamiento con las columnas a procesar.
            
        Raises:
            ValueError: Si df_train está vacío o le faltan columnas requeridas.
        """
        
        if df_train.empty:
            raise ValueError("df_train no puede estar vacío")
        
        # Comprobar si faltan columnas definidas en la inicialización
        required_cols = self.cols_to_process
        missing_cols = set(required_cols) - set(df_train.columns)
        if missing_cols:
            raise ValueError(f"A df_train le faltan las columnas: {missing_cols}")
        logger.info("Ajustando codificadores con datos de entrenamiento...")
        
        for col in self.cols_to_process:
            if col in self.multi_label_cols:
                # MultiLabelBinarizer para columnas con múltiples etiquetas.
                encoder = MultiLabelBinarizer()
                encoder.fit(df_train[col])
                self.encoders[col] = encoder
                logger.debug(f"Ajustado MultiLabelBinarizer para la columna: {col}")
                continue
            
                # LabelEncoder para todas las demás columnas (inputs y targets).
            encoder = LabelEncoder()
                
            # Optimización: obtener valores únicos directamente y añadir token desconocido
            unique_labels = df_train[col].astype(str).unique().tolist()
            unique_labels.append(self.unknown_token)
                
            encoder.fit(unique_labels)
            self.encoders[col] = encoder
            logger.debug(f"Ajustado LabelEncoder para la columna: {col}")
        
        logger.info("Encoders ajustados correctamente.")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma un DataFrame usando los codificadores ya ajustados.
        Maneja las etiquetas no vistas reemplazándolas con un token de 'desconocido'.
        
        Args:
            df: DataFrame a transformar
            
        Returns:
            DataFrame transformado con columnas codificadas
        """
        if not self.encoders:
            raise RuntimeError("fit() debe ser llamado antes de transform()")
        
        df_transformed = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df_transformed.columns:
                logging.warning(f"La columna {col} no se encontró en el DataFrame, se omite.")
                continue

            if isinstance(encoder, MultiLabelBinarizer):
                # MLB transforma las listas en vectores binarios y maneja desconocidos.
                df_transformed[col] = list(encoder.transform(df_transformed[col]))
            
            elif isinstance(encoder, LabelEncoder):
                # Optimización: operación vectorizada para reemplazar valores desconocidos
                known_labels = set(encoder.classes_)
                
                # Convertir a string y reemplazar valores desconocidos (operación vectorizada)
                col_values = df_transformed[col].astype(str)
                mask_unknown = ~col_values.isin(known_labels)
                col_values[mask_unknown] = self.unknown_token
                
                # Transformar de forma segura
                df_transformed[col] = encoder.transform(col_values)
        
        logging.info(f"Transformación completada para {len(df)} filas.")
        return df_transformed
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"PGenDataProcess("
            f"cols_to_process={len(self.cols_to_process)}, "
            f"feature_cols={len(self.cols_to_process) - len(self.target_cols)}, "
            f"target_cols={self.target_cols}, "
            f"multi_label_cols={self.multi_label_cols}, "
            f"encoders_fitted={len(self.encoders)}/{len(self.cols_to_process)})"
        )

    def _transform_single_input(self, df: pd.DataFrame) -> dict:
        ...

class PGenDataset(Dataset):
    """
    Dataset de PyTorch optimizado para datos farmacogenéticos.
    
    Convierte datos pre-procesados a tensores para entrenamiento eficiente.
    Soporta columnas multi-etiqueta y single-etiqueta.
    
    Args:
        df: DataFrame con datos procesados (post-transform)
        target_cols: Lista de nombres de columnas objetivo
        multi_label_cols: Set de columnas multi-etiqueta (opcional)
    """
    
    def __init__(self, df, feature_cols, target_cols, multi_label_cols=None):
        self.tensors = {}
        self.feature_cols = [f.lower() for f in feature_cols]
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = multi_label_cols or set()

        # Optimización: procesar solo columnas relevantes
        relevant_cols = df.columns - set(df['stratify_col'])
        
        for col in relevant_cols:
            if col in self.multi_label_cols:
                # Para columnas multi-etiqueta
                try:
                    stacked_data = np.stack(df[col].values)
                    self.tensors[col] = torch.from_numpy(stacked_data).float()
                except ValueError as e:
                    raise ValueError(
                        f"Error al apilar la columna multi-etiqueta '{col}'. "
                        "Verifica que transform() se ejecutó correctamente."
                    ) from e
            else:
                # Para inputs y targets de etiqueta única
                # Optimización: usar from_numpy para evitar copia innecesaria
                try:
                    col_values = df[col].values
                    self.tensors[col] = torch.from_numpy(col_values).long()
                except (TypeError, ValueError) as e:
                    raise TypeError(
                        f"Error al crear tensor para la columna '{col}'. "
                        "Verifica que no contenga tipos mixtos."
                    ) from e

    def __len__(self):
        """Retorna el número de muestras en el dataset."""
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, idx):
        """
        Obtiene una muestra del dataset.
        
        Args:
            idx: Índice de la muestra
            
        Returns:
            Diccionario con tensores de la muestra
        """
        return {k: v[idx] for k, v in self.tensors.items()}
