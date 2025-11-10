import glob
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import numpy as np
import pandas as pd
import src.config.config as cfg
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from src.config.config import *
from torch.utils.data import Dataset
from .model_configs import MULTI_LABEL_COLUMN_NAMES

logger = logging.getLogger(__name__)

# Constantes de configuración
MIN_SAMPLES_FOR_CLASS_GROUPING = 20  # Umbral mínimo de muestras por clase antes de agrupar


def train_data_import(targets):
    csv_path = MODEL_TRAIN_DATA
    
    
    csv_files = glob.glob(f"{csv_path}/*.tsv")

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos TSV en: {csv_path}")
    elif len(csv_files) == 1:
        logging.info(f"Se encontró un único archivo TSV. Usando: {csv_files[0]}")
        csv_files = csv_files[0]
    csv_files = Path(csv_path, "final_test_genalle.tsv")
            
    targets = [t.lower() for t in targets]

    read_cols_set = set(targets) | {"Drug", "Genalle", "Gene", "Allele"} #Variant/Haplotypes cambiado a Genalle
    read_cols = [c.lower() for c in read_cols_set]

    # equivalencias = load_equivalencias(csv_path)
    return csv_files, read_cols

def load_equivalencias(csv_path: str) -> Dict:
    """
    Load and filter genotype-allele equivalences from JSON file.
    
    Keeps only entries with RS identifiers (rs[0-9]+), removes others.
    
    Args:
        csv_path: Path to directory containing json_dicts/geno_alleles_dict.json
        
    Returns:
        Dictionary with filtered equivalences (only RS identifiers)
    """
    with open(f"{csv_path}/json_dicts/geno_alleles_dict.json", "r") as f:
        equivalencias = json.load(f)

    # Filter: keep only RS identifiers, mark others as NaN
    filtered_equivalencias = {}
    for k, v in equivalencias.items():
        if re.match(r"rs[0-9]+", k):
            filtered_equivalencias[k] = v
        else:
            filtered_equivalencias[k] = np.nan

    # Remove entries with NaN values
    filtered_equivalencias = {k: v for k, v in filtered_equivalencias.items() if not pd.isna(v)}

    logger.info(f"Loaded {len(filtered_equivalencias)} valid RS identifiers from equivalences")
    return filtered_equivalencias


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
        self.cols_to_process = []
        self.target_cols = []
        self.multi_label_cols = set()
        self.input_cols = ["drug", "genalle", "gene", "allele"]

        self.unknown_token = "__UNKNOWN__"
        self.one_hot_encoded_cols = []

    def _split_labels(self, label_str):
        """Helper: Divide un string 'A, B' en una lista ['A', 'B'] y maneja nulos."""
        if not isinstance(label_str, str) or pd.isna(label_str) or label_str.lower() in ["nan", "", "null"]:
            return []  # MultiLabelBinarizer necesita un iterable
        return [s.strip() for s in re.split(r",|;", label_str) if s.strip()]

    def load_data(self, csv_path: str, all_cols: List[str], input_cols: List[str],
                  target_cols: List[str], multi_label_targets: Optional[List[str]], 
                  stratify_cols: List[str]) -> pd.DataFrame:
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
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(str(csv_path), sep="\t", index_col=False)
        df.columns = [col.lower() for col in df.columns]
        
        # Validate that all required columns exist
        missing_cols = set(self.cols_to_process) - set(df.columns)
        if missing_cols:
            raise KeyError(f"Missing columns in CSV: {missing_cols}")
        
        df = df[self.cols_to_process]
        
        logging.info("Limpiando nombres de entidades (reemplazando espacios con '_')...")
        # Optimized: process single-label and multi-label columns separately
        single_label_cols = [col for col in self.cols_to_process if col in df.columns and col not in self.multi_label_cols]
        multi_label_cols_present = [col for col in self.multi_label_cols if col in df.columns]
        
        # Vectorized operation for all single-label columns at once
        if single_label_cols:
            for col in single_label_cols:
                df[col] = df[col].astype(str).str.replace(' ', '_', regex=False)
        
        # Process multi-label columns only if present
        if multi_label_cols_present:
            for col in multi_label_cols_present:
                df[col] = df[col].apply(
                    lambda lst: [s.replace(' ', '_') for s in lst] if isinstance(lst, list) else lst
                )
        
        task3_name = "effect_type"
        if task3_name in df.columns:
            logging.info(f"Agrupando clases raras para '{task3_name}'...")
            
            # 1. Obtener los conteos de clase
            counts = df[task3_name].value_counts()
            
            # 2. Identificar las clases a agrupar
            to_group = counts[counts < MIN_SAMPLES_FOR_CLASS_GROUPING].index
            
            if len(to_group) > 0:
                logging.info(f"Se agruparán {len(to_group)} clases en 'Other_Grouped'.")
                # Optimización: usar replace() en lugar de apply() para mejor performance
                df[task3_name] = df[task3_name].replace(to_group.tolist(), "Other_Grouped")
            else:
                logging.info("No se encontraron clases raras para agrupar.")
                
        # 1. Normaliza columnas target (y de estratificación) a string
        for t in self.target_cols:
            df[t] = df[t].astype(str)
            
        for s_col in stratify_cols:
             df[s_col] = df[s_col].astype(str)

        # 2. Crear 'stratify_col'
        # Usamos las columnas de estratificación proporcionadas
        df["stratify_col"] = df[stratify_cols].agg("_".join, axis=1)

        # 3. Filtrar datos insuficientes (opcional, pero buena práctica)
        counts = df["stratify_col"].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df["stratify_col"].isin(suficientes)].reset_index(drop=True)

        # 4. Procesar las columnas multi-etiqueta (convertir string a lista)
        for col in self.multi_label_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._split_labels)
        
        logging.info(f"Datos cargados y limpios. Total de filas: {len(df)}")
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
        
        logging.info("Ajustando codificadores con datos de entrenamiento...")
        
        # No se necesita copia completa del DataFrame, trabajamos directamente
        for col in self.cols_to_process:
            if col in self.multi_label_cols:
                # MultiLabelBinarizer para columnas con múltiples etiquetas.
                encoder = MultiLabelBinarizer()
                encoder.fit(df_train[col])
                self.encoders[col] = encoder
                logging.debug(f"Ajustado MultiLabelBinarizer para la columna: {col}")

            else:
                # LabelEncoder para todas las demás columnas (inputs y targets).
                encoder = LabelEncoder()
                
                # Optimización: obtener valores únicos directamente y añadir token desconocido
                unique_labels = df_train[col].astype(str).unique().tolist()
                unique_labels.append(self.unknown_token)
                
                encoder.fit(unique_labels)
                self.encoders[col] = encoder
                logging.debug(f"Ajustado LabelEncoder para la columna: {col}")
        
        logging.info("Codificadores ajustados correctamente.")
    
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
            f"target_cols={self.target_cols}, "
            f"multi_label_cols={self.multi_label_cols}, "
            f"encoders_fitted={len(self.encoders)}/{len(self.cols_to_process)})"
        )



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
    
    def __init__(self, df, target_cols, multi_label_cols=None):
        self.tensors = {}
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = multi_label_cols or set()

        # Optimización: procesar solo columnas relevantes
        relevant_cols = [col for col in df.columns if col != "stratify_col"]
        
        for col in relevant_cols:
            if col in self.multi_label_cols:
                # Para columnas multi-etiqueta (que contienen arrays)
                # Optimización: usar from_numpy cuando es posible para mejor performance
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


