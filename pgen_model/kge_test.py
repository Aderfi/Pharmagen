#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCRIPT CONSOLIDADO para entrenamiento, optimización y pipeline
de modelos farmacogenéticos (DeepFM_PGenModel).

Este archivo combina los siguientes módulos:
- data.py (Procesamiento de datos y Dataset)
- train_utils.py (Funciones auxiliares de entrenamiento)
- model.py (La clase DeepFM_PGenModel)
- train.py (Lógica de entrenamiento y guardado)
- pipeline.py (Pipeline de entrenamiento final)
- optuna_train.py (Lógica de optimización con Optuna)

Modificaciones realizadas:
- Añadido el método 'load_from_gensim_kv' al modelo.
- Actualizado 'train_pipeline' para cargar embeddings desde un archivo .kv.
- Corregida la importación faltante de 'FocalLoss'.
- Corregido 'device' indefinido en 'load_pretrained_embeddings'.
"""

# ============================================================================
# 1. IMPORTACIONES GLOBALES (Consolidadas)
# ============================================================================

import datetime
import glob
import itertools  # Necesario para DeepFM_PGenModel
import json
import logging
import math
import multiprocessing
import os
import pickle
import random
import re
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, overload

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_pareto_front
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OneHotEncoder
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import kge_test_packages.model_configs
from kge_test_packages.model_configs import (
        CLINICAL_PRIORITIES,
        MODEL_REGISTRY,
        MULTI_LABEL_COLUMN_NAMES,
        get_model_config,
    )

import kge_test_packages.focal_loss
from kge_test_packages.focal_loss import FocalLoss


# Importaciones externas (asumimos que existen en el entorno del usuario)
# Si estos fallan, el usuario debe crear/proporcionar estos archivos

PROJECT_ROOT = Path(".")
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_ENCODERS_DIR = PROJECT_ROOT / "encoders"
MODEL_TRAIN_DATA = PROJECT_ROOT / "train_data"
PGEN_MODEL_DIR = MODELS_DIR / "pgen"




# Importación de Gensim para la carga de .KV
from gensim.models import KeyedVectors



# Importación de FocalLoss (archivo no proporcionado)



# ============================================================================
# 2. CONFIGURACIÓN GLOBAL (Logger, Constantes)
# ============================================================================

# Configurar un logger global
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constantes de los archivos
MULTILABEL_THRESHOLD = 0.2
EPSILON = 1e-8
EPOCHS = 100
PATIENCE = 25
MIN_SAMPLES_FOR_CLASS_GROUPING = 20
N_TRIALS = 300
EPOCH = 50
PATIENCE_OPTUNA = 12 # Renombrado para evitar colisión
RANDOM_SEED = 711
VALIDATION_SPLIT = 0.2
LABEL_SMOOTHING = 0.12
N_STARTUP_TRIALS = 15
N_EI_CANDIDATES = 24
N_PRUNER_STARTUP_TRIALS = 10
N_PRUNER_WARMUP_STEPS = 5
PRUNER_INTERVAL_STEPS = 3
CLASS_WEIGHT_LOG_SMOOTHING = 2

optuna_results = Path(PROJECT_ROOT, "optuna_outputs", "figures")
optuna_results.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 3. MÓDULO: data.py (Procesamiento de datos)
# ============================================================================

def train_data_import(targets):
    csv_path = MODEL_TRAIN_DATA
    
    
    csv_files = glob.glob(f"{csv_path}/*.tsv")
    '''
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos TSV en: {csv_path}")
    elif len(csv_files) == 1:
        logging.info(f"Se encontró un único archivo TSV. Usando: {csv_files[0]}")
        csv_files = csv_files[0]
    '''
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
            if t in df.columns:
                df[t] = df[t].astype(str)
            
        for s_col in stratify_cols:
             if s_col in df.columns:
                df[s_col] = df[s_col].astype(str)

        # 2. Crear 'stratify_col'
        # Usamos las columnas de estratificación proporcionadas
        valid_stratify_cols = [s for s in stratify_cols if s in df.columns]
        if not valid_stratify_cols:
             logger.warning("Ninguna de las stratify_cols se encontró en el DataFrame. No se estratificará.")
             df["stratify_col"] = "default"
        else:
            df["stratify_col"] = df[valid_stratify_cols].agg("_".join, axis=1)


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
            logger.warning(f"A df_train le faltan las columnas: {missing_cols}. Se omitirán del encoder.")
        
        logging.info("Ajustando codificadores con datos de entrenamiento...")
        
        # No se necesita copia completa del DataFrame, trabajamos directamente
        for col in self.cols_to_process:
            if col not in df_train.columns:
                continue # Omitir si la columna no está

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
    
    def transform(self, df_train: pd.DataFrame) -> pd.DataFrame:
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
        
        df_transformed = df_train.copy()
        
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
        
        logging.info(f"Transformación completada para {len(df_train)} filas.")
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
            if col not in df.columns:
                continue

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
        if not self.tensors:
            return 0
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


# ============================================================================
# 4. MÓDULO: train_utils.py (Funciones auxiliares)
# ============================================================================

def get_input_dims(data_loader: PGenDataProcess) -> Dict[str, int]:
    """
    Get vocabulary sizes for all input columns.
    
    Args:
        data_loader: Fitted PGenDataProcess instance with encoders
    
    Returns:
        Dictionary mapping input column names to vocabulary sizes
    
    Raises:
        KeyError: If required encoder not found
        AttributeError: If encoder missing 'classes_' attribute
    """
    input_cols = ["drug", "genalle", "gene", "allele"]
    dims = {}
    
    for col in input_cols:
        try:
            if col not in data_loader.encoders:
                raise KeyError(f"Encoder not found for input column: {col}")
            
            encoder = data_loader.encoders[col]
            if not hasattr(encoder, "classes_"):
                raise AttributeError(f"Encoder for '{col}' missing 'classes_' attribute")
            
            dims[col] = len(encoder.classes_)
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting input dimension for '{col}': {e}")
            raise

    return dims


def get_output_sizes(
    data_loader: PGenDataProcess, target_cols: List[str]
) -> List[int]:
    """
    Get vocabulary sizes for all target columns.
    
    Args:
        data_loader: Fitted PGenDataProcess instance with encoders
        target_cols: List of target column names
    
    Returns:
        List of vocabulary sizes corresponding to target_cols
    
    Raises:
        KeyError: If target encoder not found
    """
    sizes = []
    for col in target_cols:
        try:
            if col not in data_loader.encoders:
                raise KeyError(f"Encoder not found for target column: {col}")
            
            encoder = data_loader.encoders[col]
            if not hasattr(encoder, "classes_"):
                raise AttributeError(f"Encoder for '{col}' missing 'classes_' attribute")
            
            sizes.append(len(encoder.classes_))
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting output size for '{col}': {e}")
            raise

    return sizes


def calculate_task_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    target_cols: List[str],
    multi_label_cols: set,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate detailed metrics (F1, precision, recall) for each task.
    
    Handles both single-label and multi-label classification tasks.
    For single-label: uses argmax for predictions.
    For multi-label: uses sigmoid with 0.5 threshold.
    
    Args:
        model: Trained model in eval mode
        data_loader: DataLoader with validation/test data
        target_cols: List of target column names
        multi_label_cols: Set of multi-label column names
        device: Torch device (CPU or CUDA)
    
    Returns:
        Dictionary with structure:
        {
            'task_name': {
                'f1_macro': float,
                'f1_weighted': float,
                'precision_macro': float,
                'recall_macro': float
            }
        }
    """
    model.eval()

    # Store predictions and ground truth
    all_preds = {col: [] for col in target_cols}
    all_targets = {col: [] for col in target_cols}

    with torch.no_grad():
        for batch in data_loader:
            drug = batch["drug"].to(device)
            genalle = batch["genalle"].to(device)
            gene = batch["gene"].to(device)
            allele = batch["allele"].to(device)

            outputs = model(drug, genalle, gene, allele)

            for col in target_cols:
                true = batch[col].to(device)
                pred = outputs[col]

                if col in multi_label_cols:
                    # Multi-label: apply sigmoid and threshold
                    probs = torch.sigmoid(pred)
                    predicted = (probs > 0.5).float()
                else:
                    # Single-label: argmax
                    predicted = torch.argmax(pred, dim=1)

                all_preds[col].append(predicted.cpu())
                all_targets[col].append(true.cpu())

    # Calculate metrics per task
    metrics = {}
    for col in target_cols:
        preds_list = all_preds[col]
        targets_list = all_targets[col]
        
        if not preds_list or not targets_list:
            logger.warning(f"No predictions or targets found for task '{col}'. Skipping metrics.")
            metrics[col] = {
                "f1_macro": 0.0, "f1_weighted": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0
            }
            continue

        preds = torch.cat(preds_list).numpy()
        targets = torch.cat(targets_list).numpy()

        if col in multi_label_cols:
            # Multi-label: use 'samples' average
            metrics[col] = {
                "f1_samples": f1_score(targets, preds, average="samples", zero_division=0),
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
                "precision_samples": precision_score(
                    targets, preds, average="samples", zero_division=0
                ),
                "recall_samples": recall_score(
                    targets, preds, average="samples", zero_division=0
                ),
            }
        else:
            # Single-label
            metrics[col] = {
                "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
                "precision_macro": precision_score(
                    targets, preds, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    targets, preds, average="macro", zero_division=0
                ),
            }

    return metrics


def create_optimizer(model, params):
    """Create optimizer based on params configuration."""
    opt_type = params.get("optimizer_type", "adamw")
    lr = params.get("learning_rate", 1e-3) # Añadido default
    weight_decay = params.get("weight_decay", 1e-5)
    
    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(params.get("adam_beta1", 0.9), params.get("adam_beta2", 0.999))
        )
    elif opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            betas=(params.get("adam_beta1", 0.9), params.get("adam_beta2", 0.999))
        )
    elif opt_type == "sgd":
        return torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay,
            momentum=params.get("sgd_momentum", 0.9)
        )
    elif opt_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # Fallback to AdamW
        logger.warning(f"Optimizador '{opt_type}' no reconocido. Usando 'adamw'.")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_criterions(target_cols, params, class_weights_task3=None, device=torch.device("cuda")):
    """Create loss functions based on params configuration."""
    criterions_list = []
    
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions_list.append(nn.BCEWithLogitsLoss())
        
        # --- MODIFICACIÓN: Manejo de FocalLoss Faltante ---
        elif col == "effect_type" and class_weights_task3 is not None:
            if FocalLoss is not None:
                # El archivo focal_loss.py fue encontrado, usar FocalLoss
                logger.info(f"Usando FocalLoss con pesos para '{col}'.")
                gamma = params.get("focal_gamma", 2.0)
                alpha_weight = params.get("focal_alpha_weight", 1.0)
                label_smoothing = params.get("label_smoothing", 0.15)
                
                # Scale class weights by alpha_weight
                scaled_weights = class_weights_task3 * alpha_weight
                
                criterions_list.append(FocalLoss(
                    alpha=scaled_weights.to(device), # Asegurarse que está en el device
                    gamma=gamma,
                    label_smoothing=label_smoothing,
                ))
            else:
                # NOTA: El archivo focal_loss.py no se encontró.
                # Se utiliza CrossEntropyLoss con pesos como fallback.
                logger.warning(f"ADVERTENCIA: 'focal_loss.py' no encontrado. Usando CrossEntropyLoss con pesos para '{col}'.")
                criterions_list.append(nn.CrossEntropyLoss(
                    weight=class_weights_task3.to(device), # Asegurarse que está en el device
                    label_smoothing=params.get("label_smoothing", 0.1)
                ))
        # --- FIN DE LA MODIFICACIÓN ---

        else:
            # Para otras tareas single-label
            label_smoothing = params.get("label_smoothing", 0.1)
            criterions_list.append(nn.CrossEntropyLoss(label_smoothing=label_smoothing))
    
    return criterions_list


def create_scheduler(optimizer, params):
    """Create learning rate scheduler based on params configuration."""
    scheduler_type = params.get("scheduler_type", "plateau")
    
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params.get("scheduler_factor", 0.5),
            patience=params.get("scheduler_patience", 5),
            
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=params.get("epochs", 100),
            eta_min=params.get("learning_rate", 1e-4) * 0.01
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=params.get("scheduler_patience", 10),
            gamma=params.get("scheduler_factor", 0.5)
        )
    elif scheduler_type == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=params.get("scheduler_factor", 0.95)
        )
    elif scheduler_type == "none":
        return None
    else:
        logger.warning(f"Scheduler '{scheduler_type}' no reconocido. No se usará scheduler.")
        return None


# ============================================================================
# 5. MÓDULO: model.py (Clase DeepFM_PGenModel)
# ============================================================================

class DeepFM_PGenModel(nn.Module):
    """
    Generalized DeepFM model for multi-task pharmacogenomics prediction.
    
    Combines Deep learning and Factorization Machine branches with multi-head
    output for simultaneous prediction of multiple targets. Uses uncertainty
    weighting for automatic task balancing.
    
    Architecture:
        - Embedding layer: Converts categorical inputs to dense vectors
        - Deep branch: Multi-layer perceptron with attention mechanism
        - FM branch: Factorization Machine for feature interactions
        - Output heads: Task-specific prediction layers
    
    References:
        - DeepFM: He et al., 2017
        - Uncertainty Weighting: Kendall & Gal, 2017
    """
    
    # Class constants for architecture
    N_FIELDS = 4  # Drug, Gene, Allele, Genalle
    VALID_NHEADS = [8, 4, 2, 1]  # Valid attention head options
    MIN_DROPOUT = 0.0
    MAX_DROPOUT = 0.9

    def __init__(
        self,
        n_drugs: int,
        n_genalles: int,
        n_genes: int,
        n_alleles: int,
        embedding_dim: int,
        n_layers: int,
        hidden_dim: int,
        dropout_rate: float,
        target_dims: Dict[str, int],
        attention_dim_feedforward: int | None = None,
        attention_dropout: float = 0.1,
        num_attention_layers: int = 1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        activation_function: str = "gelu",
        fm_dropout: float = 0.1,
        fm_hidden_layers: int = 0,
        fm_hidden_dim: int = 256,
        embedding_dropout: float = 0.1,
        separate_embedding_dims: bool = False, # Este argumento parece no usarse
        separate_emb_dims_dict = None, # Este sí se usa
    ) -> None:
        """
        Initialize DeepFM_PGenModel.
        
        Args:
            n_drugs: Number of unique drugs in vocabulary
            n_genalles: Number of unique genotypes/alleles combinations
            n_genes: Number of unique genes
            n_alleles: Number of unique alleles
            embedding_dim: Dimension of embedding vectors
            n_layers: Number of layers in deep branch
            hidden_dim: Hidden dimension for deep layers
            dropout_rate: Dropout probability (0.0 to 0.9)
            target_dims: Dictionary mapping target names to number of classes
                         Example: {"outcome": 3, "effect_type": 5}
        
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If attention head configuration fails
        """
        super().__init__()
        
        # Validate inputs
        self._validate_inputs(
            n_drugs, n_genalles, n_genes, n_alleles,
            embedding_dim, n_layers, hidden_dim, dropout_rate, target_dims
        )
        
        self.n_fields = self.N_FIELDS
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.target_dims = target_dims
        
        # Initialize uncertainty weighting parameters
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        
        # 1. Embedding layers
        if separate_emb_dims_dict is not None:
            logger.info("Usando dimensiones de embedding separadas.")
            drug_dim = int(separate_emb_dims_dict.get("drug", embedding_dim))
            genalle_dim = int(separate_emb_dims_dict.get("genalle", embedding_dim))
            gene_dim = int(separate_emb_dims_dict.get("gene", embedding_dim))
            allele_dim = int(separate_emb_dims_dict.get("allele", embedding_dim))
        else:
            # Asegurarse de que el embedding_dim base es un entero
            drug_dim = embedding_dim
            genalle_dim = embedding_dim
            gene_dim = embedding_dim
            allele_dim = embedding_dim
        
        
        self.drug_emb = nn.Embedding(n_drugs, drug_dim)
        self.genal_emb = nn.Embedding(n_genalles, genalle_dim)
        self.gene_emb = nn.Embedding(n_genes, gene_dim)
        self.allele_emb = nn.Embedding(n_alleles, allele_dim)
            

        # 2. Activation function
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.activation_function = activation_function.lower()
        if self.activation_function == "gelu":
            self.activation = nn.GELU()
        elif self.activation_function == "relu":
            self.activation = nn.ReLU()
        elif self.activation_function == "swish":
            self.activation = nn.SiLU()  # Swish = SiLU in PyTorch
        elif self.activation_function == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.GELU()  # Default fallback
        
        
        # 3. Normalization layers
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])


        # 2. Deep branch with attention
        # ATENCIÓN: El embedding_dim global se usa para la atención,
        # esto puede fallar si 'separate_emb_dims_dict' se usa y las dims no son iguales.
        # Por simplicidad, asumimos que si se usa atención, todas las dims deben ser iguales.
        if separate_emb_dims_dict and not (drug_dim == genalle_dim == gene_dim == allele_dim):
            logger.warning("Dimensiones de embedding separadas y diferentes detectadas con capa de atención.")
            logger.warning("La capa de Atención (TransformerEncoder) espera d_model uniforme.")
            logger.warning(f"Usando embedding_dim base ({embedding_dim}) para la atención.")
            attention_d_model = embedding_dim
        else:
            # Si todas las dims son iguales (o no se separaron), usar drug_dim (o cualquiera)
            attention_d_model = drug_dim
            
        deep_input_dim = drug_dim + genalle_dim + gene_dim + allele_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # Configure attention heads
        nhead = self._get_valid_nhead(attention_d_model)
        logger.debug(f"Using {nhead} attention heads for d_model={attention_d_model}")
        
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=attention_d_model,
                nhead=nhead,
                dim_feedforward=attention_dim_feedforward if attention_dim_feedforward else hidden_dim,
                dropout=attention_dropout,
                batch_first=True,
                activation=self.activation # Usar la misma activación
            ) for _ in range(num_attention_layers)
        ])
        
        # Deep layers
        self.deep_layers = nn.ModuleList()
        # La entrada a las capas 'deep' es el output aplanado de la atención
        # El tamaño es N_FIELDS * attention_d_model
        deep_attention_output_dim = self.N_FIELDS * attention_d_model
        self.deep_layers.append(nn.Linear(deep_attention_output_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.deep_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 3. FM branch
        self.fm_dropout = nn.Dropout(fm_dropout)
        # El input de FM son los embeddings *antes* de la atención
        fm_interaction_dim = (self.N_FIELDS * (self.N_FIELDS - 1)) // 2
        
        if fm_hidden_layers > 0:
            self.fm_layers = nn.ModuleList()
            self.fm_layers.append(nn.Linear(fm_interaction_dim, fm_hidden_dim))
            for _ in range(fm_hidden_layers - 1):
                self.fm_layers.append(nn.Linear(fm_hidden_dim, fm_hidden_dim))
            fm_output_dim = fm_hidden_dim
        else:
            self.fm_layers = None
            fm_output_dim = fm_interaction_dim

        # 4. Combined output dimension
        combined_dim = hidden_dim + fm_output_dim
        
        # 5. Multi-task output heads
        self.output_heads = nn.ModuleDict()
        for target_name, n_classes in target_dims.items():
            self.output_heads[target_name] = nn.Linear(combined_dim, n_classes)
        
        logger.info(f"DeepFM_PGenModel initialized with {len(self.output_heads)} output heads")
    
    @staticmethod
    def _validate_inputs(
        n_drugs: int, n_genalles: int, n_genes: int, n_alleles: int,
        embedding_dim: int, n_layers: int, hidden_dim: int,
        dropout_rate: float, target_dims: Dict[str, int]
    ) -> None:
        """Validate all input parameters."""
        if n_drugs <= 0 or n_genalles <= 0 or n_genes <= 0 or n_alleles <= 0:
            raise ValueError("All vocabulary sizes must be positive integers")
        
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        if not (0.0 <= dropout_rate <= 0.9):
            raise ValueError(f"dropout_rate must be in [0.0, 0.9], got {dropout_rate}")
        
        if not target_dims or not isinstance(target_dims, dict):
            raise ValueError("target_dims must be a non-empty dictionary")
        
        for target_name, n_classes in target_dims.items():
            if n_classes <= 0:
                raise ValueError(f"target_dims['{target_name}'] must be positive, got {n_classes}")
    
    @staticmethod
    def _get_valid_nhead(embedding_dim: int) -> int:
        """
        Get the largest valid number of attention heads for given embedding dimension.
        
        Args:
            embedding_dim: Embedding dimension
            
        Returns:
            Valid number of attention heads
            
        Raises:
            RuntimeError: If no valid nhead found
        """
        valid_nheads = [h for h in DeepFM_PGenModel.VALID_NHEADS if embedding_dim % h == 0]
        if not valid_nheads:
            raise RuntimeError(
                f"No valid attention heads for embedding_dim={embedding_dim}. "
                f"embedding_dim must be divisible by one of {DeepFM_PGenModel.VALID_NHEADS}"
            )
        return valid_nheads[0]

    def forward(
        self,
        drug: torch.Tensor,
        genalle: torch.Tensor,
        gene: torch.Tensor,
        allele: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            drug: Tensor of drug indices, shape (batch_size,)
            genalle: Tensor of genotype indices, shape (batch_size,)
            gene: Tensor of gene indices, shape (batch_size,)
            allele: Tensor of allele indices, shape (batch_size,)
        
        Returns:
            Dictionary mapping target names to prediction tensors.
            Each tensor has shape (batch_size, n_classes) for that target.
        
        Raises:
            RuntimeError: If input tensors have incompatible shapes
        """
        # Validate input shapes
        batch_size = drug.shape[0]
        if not all(t.shape[0] == batch_size for t in [genalle, gene, allele]):
            raise RuntimeError("All input tensors must have the same batch size")

        # 1. Get embeddings
        drug_vec = self.embedding_dropout(self.drug_emb(drug))
        genal_vec = self.embedding_dropout(self.genal_emb(genalle))
        gene_vec = self.embedding_dropout(self.gene_emb(gene))
        allele_vec = self.embedding_dropout(self.allele_emb(allele))

        # 2. Deep branch with attention
        # (batch_size, n_fields, embedding_dim)
        emb_stack = torch.stack(
            [drug_vec, genal_vec, gene_vec, allele_vec],
            dim=1
        )
        
        # Apply attention
        attention_output = emb_stack
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(attention_output) # type: ignore
        
        # (batch_size, n_fields * embedding_dim)
        deep_input = attention_output.flatten(start_dim=1)
        
        # Pass through deep layers
        deep_x = deep_input
        for i, layer in enumerate(self.deep_layers):
            deep_x = layer(deep_x)
            
            # Apply normalization if specified
            if self.use_batch_norm and i < len(self.batch_norms):
                deep_x = self.batch_norms[i](deep_x)
            if self.use_layer_norm and i < len(self.layer_norms):
                deep_x = self.layer_norms[i](deep_x)
            
            deep_x = self.activation(deep_x)
            deep_x = self.dropout(deep_x)
        
        # deep_output = deep_x (ya es deep_x)
        

        # 3. FM branch
        # Usar los embeddings ANTES de la atención
        embeddings_fm = [drug_vec, genal_vec, gene_vec, allele_vec]
        fm_outputs = []
        for emb_i, emb_j in itertools.combinations(embeddings_fm, 2):
            dot_product = torch.sum(emb_i * emb_j, dim=-1, keepdim=True)
            fm_outputs.append(dot_product)
        
        fm_output = torch.cat(fm_outputs, dim=-1) # (batch_size, 6)
        fm_output = self.fm_dropout(fm_output)
        
        if self.fm_layers is not None:
            for fm_layer in self.fm_layers:
                fm_output = fm_layer(fm_output)
                fm_output = self.activation(fm_output)
                fm_output = self.fm_dropout(fm_output)
                
                
        # 4. Combine branches
        combined_vec = torch.cat([deep_x, fm_output], dim=-1)

        # 5. Multi-task predictions
        predictions = {}
        for name, head_layer in self.output_heads.items():
            predictions[name] = head_layer(combined_vec)
        
        return predictions

    def calculate_weighted_loss(
        self,
        unweighted_losses: Dict[str, torch.Tensor],
        task_priorities: Dict[str, float],
    ) -> torch.Tensor:
        """
        Calculate total weighted loss using uncertainty weighting.
        
        Implements multi-task learning with automatic task balancing using
        learnable uncertainty parameters. Formula for classification:
        L_total = Σ [ L_i * exp(-s_i) + s_i ]
        where s_i = log(σ_i²) is the learnable parameter for task i.
        
        Args:
            unweighted_losses: Dictionary mapping task names to loss tensors.
                               Example: {"outcome": tensor(0.5), "effect_type": tensor(0.2)}
            task_priorities: Dictionary mapping task names to priority weights (optional).
                               If None, all tasks weighted equally.
                               Example: {"outcome": 1.5, "effect_type": 1.0}
        
        Returns:
            Scalar tensor representing total weighted loss, ready for .backward()
        
        Raises:
            KeyError: If a task in unweighted_losses was not in target_dims during init
        """
        weighted_loss_total = 0.0

        for task_name, loss_value in unweighted_losses.items():
            # Validate task exists
            if task_name not in self.log_sigmas:
                raise KeyError(
                    f"Task '{task_name}' not found in model. "
                    f"Available tasks: {list(self.log_sigmas.keys())}"
                )
            
            # Get learnable uncertainty parameter
            s_t = self.log_sigmas[task_name]

            # Calculate dynamic weight (precision = 1/sigma^2 = exp(-s_t))
            weight = torch.exp(-s_t)

            # Apply task priorities if provided
            if task_priorities is not None and task_name in task_priorities:
                priority = task_priorities[task_name]
                prioritized_loss = loss_value * priority
                weighted_task_loss = (weight * prioritized_loss) + s_t
            else:
                weighted_task_loss = (weight * loss_value) + s_t
            
            weighted_loss_total += weighted_task_loss

        return weighted_loss_total # type: ignore
    
    def load_pretrained_embeddings(self, weights_path: str, freeze: bool = False):
            """
            Carga los embeddings pre-entrenados desde un archivo .pth
            
            Args:
                weights_path: Ruta al archivo .pth con el diccionario de tensores.
                freeze: Si es True, congela los pesos para que no se actualicen durante el entrenamiento.
            """
            import os
            if not os.path.exists(weights_path):
                logger.warning(f"No se encontró archivo de pesos en {weights_path}. Se usarán pesos aleatorios.")
                return
            
            # --- CORRECCIÓN: 'device' no estaba definido ---
            # Determinar el dispositivo automáticamente
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # --- FIN CORRECCIÓN ---

            logger.info(f"Cargando embeddings pre-entrenados desde: {weights_path}")
            try:
                # Cargar el diccionario de la CPU
                pretrained_dict = torch.load(weights_path, map_location=device)
                
                # Mapeo entre las claves del diccionario guardado y las capas del modelo
                # Clave en .pth : Atributo en self
                layer_mapping = {
                    'drug': self.drug_emb,
                    'genalle': self.genal_emb, # Nota: en tu modelo se llama genal_emb
                    'gene': self.gene_emb,
                    'allele': self.allele_emb
                }
                
                loaded_count = 0
                for key, layer in layer_mapping.items():
                    if key in pretrained_dict:
                        weights = pretrained_dict[key]
                        
                        # Verificación de dimensiones
                        model_shape = layer.weight.shape
                        input_shape = weights.shape
                        
                        if model_shape != input_shape:
                            logger.error(f"Error de dimensión en '{key}': Modelo {model_shape} vs Archivo {input_shape}")
                            continue
                            
                        # Copiar los datos
                        layer.weight.data.copy_(weights)
                        loaded_count += 1
                        
                        # Congelar si se solicita
                        if freeze:
                            layer.weight.requires_grad = False
                            logger.info(f"Capa '{key}' congelada (no-trainable).")
                    else:
                        logger.warning(f"La clave '{key}' no existe en el archivo de pesos.")
                
                logger.info(f"Se cargaron exitosamente {loaded_count} capas de embeddings.")
                
            except Exception as e:
                logger.error(f"Error crítico cargando embeddings: {e}")
                raise e

    # --- ✨ NUEVO MÉTODO: Cargar Embeddings desde Gensim .kv ---
    def load_from_gensim_kv(
        self, 
        kv_model: 'KeyedVectors',
        drug_vocab: Dict[str, int],
        gene_vocab: Dict[str, int],
        allele_vocab: Dict[str, int],
        freeze: bool = False
    ):
        """
        Carga embeddings pre-entrenados desde un objeto KeyedVectors de Gensim.
        
        Este método mapea palabras de un único modelo KV a las capas
        de embedding correctas (drug, gene, allele) usando los
        diccionarios de vocabulario proporcionados.
        
        Args:
            kv_model: El modelo KeyedVectors de Gensim (ya cargado).
            drug_vocab: Diccionario {'drug_name': index}
            gene_vocab: Diccionario {'gene_name': index}
            allele_vocab: Diccionario {'allele_name': index}
            freeze: Si es True, congela los pesos de los embeddings.
        """
        if KeyedVectors is None:
            logger.error("Gensim no está instalado. No se pueden cargar embeddings KV.")
            return

        logger.info("Iniciando carga de embeddings desde modelo Gensim KV...")

        # Mapeos de las capas del modelo y sus vocabularios
        # (Nombre de capa, capa de embedding, vocabulario)
        mappings = [
            ("drug", self.drug_emb, drug_vocab),
            ("gene", self.gene_emb, gene_vocab),
            ("allele", self.allele_emb, allele_vocab) 
            # Nota: 'genal_emb' se omite a menos que tengas un vocabulario para él.
        ]

        for layer_name, embedding_layer, vocab in mappings:
            
            # Obtener la matriz de pesos actual (inicializada aleatoriamente)
            weights = embedding_layer.weight.data
            
            found = 0
            not_found = 0
            
            logger.info(f"Procesando capa: '{layer_name}'...")
            
            # Iterar sobre el vocabulario del *modelo*
            for word, index in vocab.items():
                
                # Asegurarse de que el índice está dentro de los límites de la capa
                if index >= embedding_layer.num_embeddings:
                    logger.warning(f"Índice {index} para '{word}' en '{layer_name}' está fuera de rango ({embedding_layer.num_embeddings}). Omitiendo.")
                    continue
                    
                try:
                    # 1. Buscar la palabra en el modelo Gensim KV
                    vector = kv_model[word]
                    
                    # 2. Convertir el vector (numpy) a tensor de PyTorch
                    vector_tensor = torch.tensor(vector, dtype=weights.dtype)
                    
                    # 3. Verificar que la dimensión coincida
                    if vector_tensor.shape[0] != weights.shape[1]:
                        logger.error(
                            f"Error de dimensión en '{layer_name}' para '{word}': "
                            f"Modelo espera {weights.shape[1]}, KV tiene {vector_tensor.shape[0]}. Omitiendo."
                        )
                        continue
                        
                    # 4. Copiar el vector pre-entrenado en la fila correcta de la matriz de pesos
                    weights[index, :] = vector_tensor
                    found += 1
                    
                except KeyError:
                    # La palabra del vocabulario no se encontró en el archivo KV
                    not_found += 1
                except Exception as e:
                    logger.error(f"Error procesando '{word}' en capa '{layer_name}': {e}")
            
            logger.info(f"Capa '{layer_name}': {found} vectores cargados, {not_found} no encontrados (se mantienen aleatorios).")
            
            # 5. Asignar la nueva matriz de pesos actualizada a la capa
            embedding_layer.weight.data.copy_(weights)
            
            # 6. Congelar la capa si se solicita
            if freeze:
                embedding_layer.weight.requires_grad = False
                logger.info(f"Capa '{layer_name}' congelada (no-trainable).")
        
        logger.info("Carga de embeddings desde Gensim KV completada.")
    # --- FIN DEL NUEVO MÉTODO ---
    
    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"DeepFM_PGenModel("
            f"embedding_dim={self.embedding_dim}, "
            f"n_layers={self.n_layers}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dropout_rate={self.dropout_rate}, "
            f"n_tasks={len(self.output_heads)}, "
            f"tasks={list(self.target_dims.keys())})"
        )


# ============================================================================
# 6. MÓDULO: train.py (Lógica de entrenamiento)
# ============================================================================

# Overloads para train_model
@overload
def train_model( # type: ignore
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Union[nn.Module, torch.optim.Optimizer]],
    epochs: int,
    patience: int,
    model_name: str,
    device: Optional[torch.device] = None,
    target_cols: Optional[List[str]] = None,
    scheduler: Optional[Any] = None,
    params_to_txt: Optional[Dict[str, Any]] = None,
    multi_label_cols: Optional[set] = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: Optional[Dict[str, float]] = None,
    return_per_task_losses: bool = True,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[float, List[float], List[float]]:
    """Overload: return_per_task_losses=True returns 3 values."""
    ...


@overload
def train_model( # type:ignore
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Union[nn.Module, torch.optim.Optimizer]],
    epochs: int,
    patience: int,
    model_name: str,
    device: Optional[torch.device] = None,
    target_cols: Optional[List[str]] = None,
    scheduler: Optional[Any] = None,
    params_to_txt: Optional[Dict[str, Any]] = None,
    multi_label_cols: Optional[set] = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: Optional[Dict[str, float]] = None,
    return_per_task_losses: bool = False,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[float, List[float]]:
    """Overload: return_per_task_losses=False returns 2 values."""
    ...


# Función train_model principal
def train_model(
    train_loader: Any,
    val_loader: Any,
    model: DeepFM_PGenModel,
    criterions: List[Union[nn.Module, torch.optim.Optimizer]],
    epochs: int,
    patience: int,
    model_name: str,
    device: Optional[torch.device] = None,
    target_cols: Optional[List[str]] = None,
    scheduler: Optional[Any] = None,
    params_to_txt: Optional[Dict[str, Any]] = None,
    multi_label_cols: Optional[set] = None,
    progress_bar: bool = False,
    optuna_check_weights: bool = False,
    use_weighted_loss: bool = False,
    task_priorities: Optional[Dict[str, float]] = None,
    return_per_task_losses: bool = False,
    trial: Optional[optuna.Trial] = None,
) -> Union[Tuple[float, List[float]], Tuple[float, List[float], List[float]]]:
    """
    Train a multi-task deep learning model with early stopping.
    
    Implements the main training loop with support for multi-task learning,
    early stopping, Optuna integration, and comprehensive metric tracking.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        model: DeepFM_PGenModel instance to train
        criterions: List of loss functions followed by optimizer
                   Format: [loss_fn_1, loss_fn_2, ..., optimizer]
        epochs: Maximum number of training epochs
        patience: Number of epochs without improvement before early stopping
        model_name: Name of the model (for logging and saving)
        device: Torch device (CPU or CUDA). If None, auto-detects.
        target_cols: List of target column names. Required.
        scheduler: Optional learning rate scheduler
        params_to_txt: Optional dict of hyperparameters to save
        multi_label_cols: Set of multi-label column names
        progress_bar: If True, show progress bars
        optuna_check_weights: If True, print per-task losses and exit
        use_weighted_loss: If True, use uncertainty weighting. If False, sum losses.
        task_priorities: Optional dict of task priority weights
        return_per_task_losses: If True, return per-task losses as 3rd value
        trial: Optional Optuna trial for pruning
    
    Returns:
        If return_per_task_losses=True:
            (best_loss, best_accuracies, per_task_losses)
        If return_per_task_losses=False:
            (best_loss, best_accuracies)
    
    Raises:
        ValueError: If target_cols not provided or criterions mismatch
        RuntimeError: If model training fails
    """
    # Validate inputs
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device auto-selected: {device}")

    if target_cols is None:
        raise ValueError("target_cols must be provided as a list of target column names")

    if not isinstance(target_cols, list) or len(target_cols) == 0:
        raise ValueError("target_cols must be a non-empty list")

    if multi_label_cols is None:
        multi_label_cols = set()

    num_targets = len(target_cols)

    # Extract optimizer and loss functions
    optimizer = criterions[-1]
    criterions_list = criterions[:-1]

    if len(criterions_list) != num_targets:
        raise ValueError(
            f"Mismatch: received {len(criterions_list)} loss functions, "
            f"but expected {num_targets} (based on target_cols)"
        )

    # Initialize tracking variables
    best_loss = float("inf")
    best_accuracies = [0.0] * num_targets
    trigger_times = 0
    individual_loss_sums = [0.0] * num_targets

    model = model.to(device)
    logger.info(f"Model moved to device: {device}")
    
    # Initialize mixed precision training (AMP) for faster training on GPU
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    if use_amp:
        logger.info("Mixed precision training (AMP) enabled for faster GPU computation")

    # Training loop
    epoch_iterator = (
        tqdm(range(epochs), desc="Training", unit="epoch")
        if progress_bar
        else range(epochs)
    )

    for epoch in epoch_iterator:
        # ====================================================================
        # Training Phase
        # ====================================================================
        model.train()
        total_loss = 0.0

        train_iterator = (
            tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}",
                unit="batch",
                colour="green",
                leave=False,
            )
            if progress_bar
            else train_loader
        )

        for batch in train_iterator:
            # Move inputs to device (non_blocking for faster transfer)
            drug = batch["drug"].to(device, non_blocking=True)
            genalle = batch["genalle"].to(device, non_blocking=True)
            gene = batch["gene"].to(device, non_blocking=True)
            allele = batch["allele"].to(device, non_blocking=True)

            # Move targets to device
            targets = {col: batch[col].to(device, non_blocking=True) for col in target_cols}

            # Forward pass with mixed precision
            optimizer.zero_grad()
            
            # --- Cálculo de Loss (unificado) ---
            def calculate_losses():
                outputs = model(drug, genalle, gene, allele)
                
                # Calcular pérdidas individuales (no ponderadas)
                individual_losses_dict = {}
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_list[i]
                    pred = outputs[col]
                    true = targets[col]
                    individual_losses_dict[col] = loss_fn(pred, true) # type: ignore
                
                # Combinar pérdidas
                if use_weighted_loss and hasattr(model, 'calculate_weighted_loss'):
                    # Usar Ponderación de Incertidumbre (Uncertainty Weighting)
                    loss = model.calculate_weighted_loss(
                        individual_losses_dict,
                        task_priorities # type: ignore
                    )
                else:
                    # Suma simple
                    loss = torch.stack(list(individual_losses_dict.values())).sum()
                
                return loss, outputs # Devuelve outputs para el raro caso de AMP-else
            # --- Fin de cálculo de Loss ---


            if use_amp and scaler is not None:
                with autocast(device_type=str(device).split(":")[0]): # 'cuda' o 'cpu'
                    loss, _ = calculate_losses()
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward() # type: ignore
                scaler.step(optimizer) # type: ignore
                scaler.update() # type: ignore
            else:
                loss, outputs = calculate_losses()
                
                # Backward pass
                loss.backward()
                optimizer.step() # type: ignore

            total_loss += loss.item()

            if progress_bar:
                train_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})

        # ====================================================================
        # Validation Phase
        # ====================================================================
        model.eval()
        val_loss = 0.0
        corrects = [0] * num_targets
        totals = [0] * num_targets
        individual_loss_sums = [0.0] * num_targets

        with torch.no_grad():
            for batch in val_loader:
                # Move inputs to device (non_blocking for faster transfer)
                drug = batch["drug"].to(device, non_blocking=True)
                genalle = batch["genalle"].to(device, non_blocking=True)
                gene = batch["gene"].to(device, non_blocking=True)
                allele = batch["allele"].to(device, non_blocking=True)

                # Move targets to device
                targets = {col: batch[col].to(device, non_blocking=True) for col in target_cols}

                # Forward pass
                outputs = model(drug, genalle, gene, allele)

                # Calcular pérdidas individuales (no ponderadas)
                individual_losses_val = []
                individual_losses_val_dict = {}
                for i, col in enumerate(target_cols):
                    loss_fn = criterions_list[i]
                    pred = outputs[col]
                    true = targets[col]
                    l = loss_fn(pred, true) # type: ignore
                    individual_losses_val.append(l)
                    individual_losses_val_dict[col] = l

                # Track individual losses (unweighted)
                individual_losses_val_tensor = torch.stack(individual_losses_val)
                for i in range(num_targets):
                    individual_loss_sums[i] += individual_losses_val_tensor[i].item()

                # Combinar pérdidas
                if use_weighted_loss and hasattr(model, 'calculate_weighted_loss'):
                    loss = model.calculate_weighted_loss(
                        individual_losses_val_dict,
                        task_priorities # type: ignore
                    )
                else:
                    loss = torch.stack(individual_losses_val).sum()
                
                val_loss += loss.item()

                # Calculate per-task accuracies
                for i, col in enumerate(target_cols):
                    pred = outputs[col]
                    true = targets[col]

                    if col in multi_label_cols:
                        # Multi-label: Hamming accuracy
                        probs = torch.sigmoid(pred)
                        predicted = (probs > MULTILABEL_THRESHOLD).float()
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.numel()
                    else:
                        # Single-label: standard accuracy
                        _, predicted = torch.max(pred, 1)
                        corrects[i] += (predicted == true).sum().item()
                        totals[i] += true.size(0)

        # Normalize validation loss
        val_loss /= len(val_loader)

        # Optuna pruning (if applicable)
        if trial is not None:
            try:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    logger.info(f"Trial pruned at epoch {epoch} (val_loss={val_loss:.4f})")
                    raise optuna.TrialPruned()
            except NotImplementedError:
                # Multi-objective optimization doesn't support pruning
                pass

        # Debug mode: print per-task losses and exit
        if optuna_check_weights:
            avg_individual_losses = [
                loss_sum / len(val_loader) for loss_sum in individual_loss_sums
            ]
            logger.info("=" * 60)
            logger.info("Epoch Validation Summary")
            logger.info("=" * 60)
            logger.info(f"Total Weighted Val Loss: {val_loss:.5f}")
            logger.info("Average Individual Task Losses (Unweighted):")
            for i, col in enumerate(target_cols):
                logger.info(f"  {col}: {avg_individual_losses[i]:.5f}")
            logger.info("=" * 60)
            raise SystemExit(0)

        # Learning rate scheduling
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Calculate accuracies
        val_accuracies = [
            c / t if t > EPSILON else 0.0 for c, t in zip(corrects, totals)
        ]
        
        # Log de época
        acc_log = ", ".join([f"{col}: {acc:.3f}" for col, acc in zip(target_cols, val_accuracies)])
        epoch_log_msg = f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Accs: [{acc_log}]"
        
        if progress_bar and isinstance(epoch_iterator, tqdm):
            epoch_iterator.set_postfix_str(epoch_log_msg)
        else:
            logger.info(epoch_log_msg)


        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_accuracies = val_accuracies.copy()
            trigger_times = 0
            logger.debug(
                f"Epoch {epoch}: New best loss {best_loss:.5f}, "
                f"accuracies {best_accuracies}"
            )
            # Guardar el mejor modelo temporalmente (opcional)
            # torch.save(model.state_dict(), "temp_best_model.pth")
        else:
            trigger_times += 1
            if trigger_times >= patience:
                logger.info(
                    f"Early stopping triggered en época {epoch + 1} "
                    f"tras {patience} épocas sin mejora."
                )
                break

    # Return results based on flag
    # Calcular las pérdidas promedio de la *mejor* época (o la última)
    avg_per_task_losses = [
        loss_sum / len(val_loader) for loss_sum in individual_loss_sums
    ]

    if return_per_task_losses:
        return best_loss, best_accuracies, avg_per_task_losses
    else:
        return best_loss, best_accuracies


# ============================================================================
# Model Saving Function
# ============================================================================


def save_model(
    model: DeepFM_PGenModel,
    target_cols: List[str],
    best_loss: float,
    best_accuracies: List[float],
    model_name: str,
    avg_per_task_losses: List[float],
    params_to_txt: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save trained model and generate report.
    
    Saves model weights to .pth file and generates a text report with
    training metrics and hyperparameters.
    
    Args:
        model: Trained DeepFM_PGenModel instance
        target_cols: List of target column names
        best_loss: Best validation loss achieved
        best_accuracies: List of best accuracies per task
        model_name: Name of the model
        avg_per_task_losses: List of average per-task losses
        params_to_txt: Optional dict of hyperparameters to save
    
    Raises:
        IOError: If model or report cannot be saved
    """
    try:
        # Create model directory
        model_save_dir = Path(MODELS_DIR)
        model_save_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        path_model_file = model_save_dir / f"pmodel_{model_name}.pth"
        torch.save(model.state_dict(), path_model_file)
        logger.info(f"Model weights saved: {path_model_file}")
        
        # Save model (pickle - aunque torch.save de state_dict es preferido)
        model_pickle_file = model_save_dir / f"pickledpmodel_{model_name}.pkl"
        try:
             with open(model_pickle_file, 'wb') as f:
                 pickle.dump(model, f)
             logger.info(f"Modelo completo (pickle) guardado: {model_pickle_file}")
        except Exception as e:
             logger.warning(f"No se pudo guardar el modelo completo con pickle: {e}")
        
        # Create report directory
        path_txt_file = model_save_dir / "txt_files"
        path_txt_file.mkdir(parents=True, exist_ok=True)

        # Generate report filename
        report_filename = f"report_{model_name}_{round(best_loss, 5)}.txt"
        file_report = path_txt_file / report_filename

        # Write report
        with open(file_report, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("MODEL TRAINING REPORT\n")
            f.write("=" * 70 + "\n\n")

            f.write("Model Configuration:\n")
            f.write(f"  Model Name: {model_name}\n")
            f.write(f"  Target Columns: {', '.join(target_cols)}\n\n")

            f.write("Training Results:\n")
            f.write(f"  Best Validation Loss: {best_loss:.5f}\n\n")

            f.write("Per-Task Average Losses (en la última época):\n")
            for i, col in enumerate(target_cols):
                f.write(f"  {col}: {avg_per_task_losses[i]:.5f}\n")
            f.write("\n")

            f.write("Per-Task Best Accuracies (en la mejor época de loss):\n")
            for i, col in enumerate(target_cols):
                f.write(f"  {col}: {best_accuracies[i]:.4f}\n")
            f.write("\n")

            if params_to_txt:
                f.write("Hyperparameters:\n")
                for key, val in params_to_txt.items():
                    f.write(f"  {key}: {val}\n")
            else:
                f.write("Hyperparameters: Not available\n")

            f.write("\n" + "=" * 70 + "\n")

        logger.info(f"Report saved: {file_report}")

    except IOError as e:
        logger.error(f"Failed to save model or report: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during model saving: {e}")
        raise


# ============================================================================
# 7. MÓDULO: pipeline.py (Pipeline de entrenamiento final)
# ============================================================================

def train_pipeline(
    PGEN_MODEL_DIR_STR: str, # Renombrado para evitar conflicto con la variable global
    csv_files: Any, # Este argumento parece no usarse, pero lo mantenemos
    model_name: str,
    target_cols: Optional[List[str]] = None,
    patience: int = PATIENCE,
    epochs: int = EPOCHS,
    # El argumento 'params' que se pasaba en __main__ ahora se lee desde config
):
    """
    Función principal para entrenar un modelo PGen.
    """
    seed = 711  # Mantener una semilla fija para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Para GPU
        # Enable cuDNN benchmarking for improved performance with fixed input sizes
        torch.backends.cudnn.benchmark = True
        # Enable TF32 on Ampere GPUs for faster matrix operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Obtener Configuración Completa del Modelo ---
    try:
        config = get_model_config(model_name)
    except Exception as e:
        logger.error(f"No se pudo cargar la configuración para el modelo '{model_name}'. Verifica 'model_configs.py'.")
        logger.error(e)
        return None # Salir de la pipeline

    inputs_from_config = config["inputs"]  # Columnas de entrada definidas para el modelo
    cols_from_config = config["cols"]  # Columnas a leer del CSV
    targets_from_config = config["targets"]  # Targets definidos para el modelo
    
    # Usar 'effect_type' que es single-label para estratificación
    stratify_cols = ['effect_type']
    
    # Extraer hiperparámetros del diccionario 'params' de la config
    params = config.get("params", {})
    if not params:
        logger.warning(f"No se encontraron 'params' en la config de '{model_name}'. Se usarán defaults.")
        
    epochs = params.get("epochs", epochs) # Usar valor de param o el default de la función
    patience = params.get("patience", patience)
    batch_size = params.get("batch_size", 64) # Default si no está en params
    
    # --------------------------------------------------

    # Determinar las columnas target finales a usar
    if target_cols is None:
        target_cols = [t.lower() for t in targets_from_config]
    else:
        target_cols = [t.lower() for t in target_cols]

    # --- Cargar y Preparar Datos ---
    # 'csv_files' se pasa como argumento, pero 'train_data_import' parece mejor
    try:
        actual_csv_files, _ = train_data_import(targets_from_config)
    except FileNotFoundError as e:
        logger.error(e)
        return None

    data_loader_obj = PGenDataProcess()  # 1. Crear el procesador

    # 2. Cargar y limpiar el DataFrame COMPLETO
    try:
        df = data_loader_obj.load_data(
            str(actual_csv_files), # Asegurarse que es string
            cols_from_config,
            inputs_from_config,
            targets_from_config,
            multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
            stratify_cols=stratify_cols,
        )
    except (AttributeError, KeyError, FileNotFoundError) as e:
        logger.error(f"Error cargando los datos: {e}")
        raise

    logger.info(f"Semilla utilizada: {seed}")
    
    # 3. Dividir ANTES de procesar (Usando estratificación)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2, # Fracción para validación
        random_state=seed,
        stratify=df["stratify_col"] # <-- Ahora usa 'effect_type'
    )
    val_df = val_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    logger.info(f"División estratificada completada. Train: {len(train_df)}, Val: {len(val_df)}")

    # 4. Ajustar (FIT) SÓLO con datos de entrenamiento
    data_loader_obj.fit(train_df)
    
    # 5. Transformar AMBOS sets por separado
    train_processed_df = data_loader_obj.transform(train_df)
    val_processed_df = data_loader_obj.transform(val_df)
    
    
    # --- Cálculo de Pesos de Clase (Class Weights) ---
    class_weights_task3 = None
    task3_name = 'effect_type'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if task3_name in target_cols:
        logger.info(f"Calculando pesos de clase SUAVES para '{task3_name}'...")
        try:
            encoder_task3 = data_loader_obj.encoders[task3_name]
            
            # Contar ocurrencias en el set de entrenamiento (ya transformado a índices)
            counts = train_processed_df[task3_name].value_counts().sort_index()
            
            all_counts = torch.zeros(len(encoder_task3.classes_))
            for class_id, count in counts.items():
                if int(class_id) < len(all_counts): # type: ignore
                    all_counts[int(class_id)] = count # type: ignore
            
            # Usar la fórmula de Log Smoothing
            weights = 1.0 / torch.log(all_counts + CLASS_WEIGHT_LOG_SMOOTHING) # +N para evitar log(1)=0 o log(0)
            weights = weights / weights.mean() # Normalizar
            class_weights_task3 = weights.to(device)
            logger.info(f"Pesos de clase listos (Min: {weights.min():.3f}, Max: {weights.max():.3f}).")
            
        except Exception as e:
            logger.error(f"Error calculando pesos de clase en pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    # --- FIN Cálculo de Pesos ---

    # Crear Datasets y DataLoaders
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    # Optimize num_workers based on available CPU cores
    num_workers = min(multiprocessing.cpu_count(), 8)
    
    # Enable pin_memory for faster GPU transfer when CUDA is available
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    # --- Preparar Información para el Modelo ---
    n_targets_list = get_output_sizes(data_loader_obj, target_cols)
    target_dims = {
        col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)
    }

    # Obtener número de clases para los inputs
    input_dims = get_input_dims(data_loader_obj)

    # --- Instanciar el Modelo ---
    model = DeepFM_PGenModel(
        n_drugs=input_dims["drug"],
        n_genalles=input_dims["genalle"],
        n_genes=input_dims["gene"],
        n_alleles=input_dims["allele"],
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        target_dims=target_dims,  # Dinámico #type: ignore
        ############
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.0),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 256),
        embedding_dropout=params.get("embedding_dropout", 0.1),
        # 'separate_emb_dims_dict' se pasa desde optuna, aquí usamos 'params'
        separate_emb_dims_dict=params.get("separate_emb_dims_dict"),
    )

    model = model.to(device)

    # --- [✨ NUEVO] Cargar Embeddings Pre-entrenados .KV ---
    # FIXME: Especifica la ruta a tu archivo .kv y si debe congelarse
    KV_FILE_PATH = "encoders/pgx_embeddings.kv" 
    KV_FREEZE_EMBEDDINGS = False # Poner en True para no re-entrenar los embeddings

    if KeyedVectors is not None and Path(KV_FILE_PATH).exists():
        logger.info(f"Cargando embeddings KV desde: {KV_FILE_PATH}")
        try:
            # Cargar el archivo .kv (asumiendo formato de texto word2vec)
            # Cambia binary=True si tu archivo .kv es binario
            kv_model = KeyedVectors.load_word2vec_format(KV_FILE_PATH, binary=False)
            
            logger.info("Creando mapeos de vocabulario desde encoders...")
            # Crear los diccionarios de vocabulario desde los encoders
            # {palabra: índice}
            # Usamos .classes_ que es una lista donde el índice es el ID
            drug_vocab = {word: i for i, word in enumerate(data_loader_obj.encoders["drug"].classes_)}
            gene_vocab = {word: i for i, word in enumerate(data_loader_obj.encoders["gene"].classes_)}
            allele_vocab = {word: i for i, word in enumerate(data_loader_obj.encoders["allele"].classes_)}
            
            logger.info(f"Vocabulario de Drugs: {len(drug_vocab)} palabras")
            logger.info(f"Vocabulario de Genes: {len(gene_vocab)} palabras")
            logger.info(f"Vocabulario de Alleles: {len(allele_vocab)} palabras")

            # Llamar al nuevo método del modelo
            model.load_from_gensim_kv(
                kv_model=kv_model,
                drug_vocab=drug_vocab,
                gene_vocab=gene_vocab,
                allele_vocab=allele_vocab,
                freeze=KV_FREEZE_EMBEDDINGS
            )
        except Exception as e:
            logger.error(f"Fallo al cargar o aplicar embeddings KV: {e}")
            logger.error("Continuando con embeddings aleatorios.")
    elif KeyedVectors is not None:
         logger.warning(f"No se cargaron embeddings KV: El archivo no existe en {KV_FILE_PATH}")
    # --- [FIN] Carga de .KV ---


    # --- Definir Optimizador y Criterios de Loss ---
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)

    criterions_list = create_criterions(target_cols, params, class_weights_task3, device)
    criterions = criterions_list + [optimizer]
    
    # Determinar si usar weighted loss
    use_weighted_loss = params.get("use_weighted_loss", True) # Activar por defecto
    task_priorities = CLINICAL_PRIORITIES if use_weighted_loss else None
    
    # ---------------------------------------------------------

    # --- Ejecutar Entrenamiento ---
    loss_mode = "Ponderación de Incertidumbre" if use_weighted_loss else "Suma Simple"
    logger.info(f"Iniciando entrenamiento final para {model_name} (Loss: {loss_mode})...")
    
    best_loss, best_accuracy_list, avg_per_task_losses = train_model( #type: ignore
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=epochs,
        patience=patience,
        model_name=model_name,
        device=device,
        target_cols=target_cols,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        params_to_txt=params,
        scheduler=scheduler,
        progress_bar=True,
        return_per_task_losses=True,
        use_weighted_loss=use_weighted_loss,
        task_priorities=task_priorities,
    )

    # --- Guardar Resultados ---
    logger.info(f"Entrenamiento completado. Mejor loss en validación: {best_loss:.5f}")
    
    save_model( model=model,
                target_cols=target_cols,
                best_loss=best_loss,
                best_accuracies=best_accuracy_list,
                avg_per_task_losses=avg_per_task_losses,
                model_name=model_name,
                params_to_txt=params
            )
    
    # Usar el path PGEN_MODEL_DIR_STR pasado como argumento
    results_dir = Path(PGEN_MODEL_DIR_STR) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = (
        results_dir / f"training_report_{model_name}_{round(best_loss, 4)}.txt"
    )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Mejor loss en validación: {best_loss:.5f}\n")
        if best_accuracy_list:
            avg_acc = sum(best_accuracy_list) / len(best_accuracy_list)
            f.write(f"Precisión Promedio (Avg Acc) en mejor época: {avg_acc:.4f}\n")
            f.write("Precisión por Tarea:\n")
            for i, col in enumerate(target_cols):
                f.write(f"  - {col}: {best_accuracy_list[i]:.4f}\n")
        if avg_per_task_losses:
            f.write("Pérdida Promedio por Tarea (última época):\n")
            for i, col in enumerate(target_cols):
                f.write(f"  - {col}: {avg_per_task_losses[i]:.4f}\n")
        
        f.write("\nHiperparámetros Utilizados:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")


    logger.info(f"Reporte de entrenamiento guardado en: {report_file}")

    # Guardar los encoders usados
    encoders_dir = Path(MODEL_ENCODERS_DIR)
    encoders_dir.mkdir(parents=True, exist_ok=True)

    encoders_file = encoders_dir / f"encoders_{model_name}.pkl"
    joblib.dump(data_loader_obj.encoders, encoders_file)
    logger.info(f"Encoders guardados en: {encoders_file}")

    return model


# ============================================================================
# 8. MÓDULO: optuna_train.py (Optimización)
# ============================================================================

def get_optuna_params(trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
    """Enhanced parameter suggestion with new search space."""
    model_conf = MODEL_REGISTRY.get(model_name)
    if not model_conf:
        # Fallback si get_model_config falló y MODEL_REGISTRY está vacío
        model_conf = get_model_config(model_name)
        if not model_conf:
             raise ValueError(f"Model configuration not found: {model_name}")

    search_space = model_conf.get("params_optuna")
    if not search_space:
        logger.warning(f"No 'params_optuna' found for '{model_name}'")
        return model_conf.get("params", {}).copy()

    suggested_params = {}

    for param_name, space_definition in search_space.items():
        try:
            if isinstance(space_definition, list):
                if space_definition and space_definition[0] == "int":
                    # Integer parameter: ["int", low, high, step(optional), log(optional)]
                    _, low, high = space_definition[:3]
                    step = space_definition[3] if len(space_definition) > 3 else 1
                    log = space_definition[4] if len(space_definition) > 4 else False
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, int(low), int(high), step=int(step), log=bool(log)
                    )
                else:
                    # Categorical parameter
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name, space_definition
                    )
            elif isinstance(space_definition, tuple):
                # Float parameter: (low, high)
                low, high = space_definition
                # Determine if log scale should be used
                is_log_scale = param_name in [
                    "learning_rate", "weight_decay", "attention_dropout",
                    "dropout_rate", "fm_dropout", "embedding_dropout"
                ]
                suggested_params[param_name] = trial.suggest_float(
                    param_name, low, high, log=is_log_scale
                )
            else:
                logger.warning(f"Unknown search space type for '{param_name}': {type(space_definition)}")
        
        except (ValueError, TypeError, IndexError) as e:
            logger.error(f"Error parsing search space for '{param_name}': {e}")

    return suggested_params


def optuna_objective(
    trial: optuna.Trial,
    model_name: str,
    pbar: tqdm,
    target_cols: List[str],
    fitted_data_loader: PGenDataProcess,
    train_processed_df: pd.DataFrame,
    val_processed_df: pd.DataFrame,
    csvfiles: Path,
    class_weights_task3: Optional[torch.Tensor] = None,
    use_multi_objective: bool = True,
) -> Tuple[float, ...]:
    """
    Optuna objective function for hyperparameter optimization.
    
    Trains a model with suggested hyperparameters and evaluates it on validation data.
    Supports both single and multi-objective optimization.
    
    Args:
        trial: Optuna trial object
        model_name: Name of the model to train
        pbar: Progress bar for updates
        target_cols: List of target column names
        fitted_data_loader: Fitted PGenDataProcess with encoders
        train_processed_df: Processed training DataFrame
        val_processed_df: Processed validation DataFrame
        csvfiles: Path to CSV files (for logging)
        class_weights_task3: Optional class weights for imbalanced tasks
        use_multi_objective: If True, optimize (loss, -F1). If False, only loss.
    
    Returns:
        Tuple of objective values:
        - If multi-objective: (loss, -critical_f1)
        - If single-objective: (loss,)
    """
    # Set random seeds for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get suggested hyperparameters
    full_base_config = get_model_config(model_name)
    base_params = full_base_config.get("params", {})
    
    # Get the parameters suggested by Optuna for this specific trial
    optuna_suggested_params = get_optuna_params(trial, model_name)
    
    # Merge them: Optuna's suggestions will overwrite the base parameters
    params = {**base_params, **optuna_suggested_params}
    
    # Save the complete and correct set of parameters to the trial for reporting
    trial.set_user_attr("full_params", params)
    

    # 2. Create datasets and dataloaders
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    # Optimize DataLoader configuration
    batch_size = params.get("batch_size", 64)
    num_workers = min(multiprocessing.cpu_count(), 8)
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    # 3. Create model
    input_dims = get_input_dims(fitted_data_loader)
    n_targets_list = get_output_sizes(fitted_data_loader, target_cols)
    target_dims = {col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)}

    separate_embedding_dims = None
    if any(key.endswith("_embedding_dim") for key in params):
        separate_embedding_dims = {
            "drug": params.get("drug_embedding_dim", params["embedding_dim"]),
            "gene": params.get("gene_embedding_dim", params["embedding_dim"]),
            "allele": params.get("allele_embedding_dim", params["embedding_dim"]),
            "genalle": params.get("genalle_embedding_dim", params["embedding_dim"]),
        }

    if trial.number == 0:
        logger.info(f"Loading data from: {Path(csvfiles)}")
        logger.info(f"Tasks: {', '.join(target_cols)}")
        logger.info(f"Clinical priorities: {CLINICAL_PRIORITIES}")
        
    model = DeepFM_PGenModel(
        n_drugs=input_dims["drug"],
        n_genalles=input_dims["genalle"],
        n_genes=input_dims["gene"],
        n_alleles=input_dims["allele"],
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        target_dims=target_dims,
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.1),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 256),
        embedding_dropout=params.get("embedding_dropout", 0.1),
        separate_emb_dims_dict=separate_embedding_dims if separate_embedding_dims != None else None,    
    ).to(device)
    
    # 4. Define optimizer and loss functions
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)
    
    criterions_list = create_criterions(target_cols, params, class_weights_task3, device)
    criterions = criterions_list + [optimizer]
    
    epochs = params.get("epochs", EPOCH)
    patience = params.get("patience", PATIENCE_OPTUNA)
    
    if "gradient_clip_norm" in params:
        torch.nn.utils.clip_grad_norm_(model.parameters(), params["gradient_clip_norm"])
    
    # Determinar si usar weighted loss
    use_weighted_loss = params.get("use_weighted_loss", True) # Activar por defecto
    task_priorities = CLINICAL_PRIORITIES if use_weighted_loss else None
        
    # 5. Train model
    best_loss, best_accuracies_list, avg_per_task_losses = train_model( # type: ignore
        train_loader, 
        val_loader, 
        model, 
        criterions,
        epochs=epochs, 
        patience=patience, 
        target_cols=target_cols,
        scheduler=scheduler, 
        params_to_txt=params,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        progress_bar=False, 
        model_name=model_name,
        trial=trial,
        use_weighted_loss=use_weighted_loss,
        task_priorities=task_priorities,
        return_per_task_losses=True,
    )
    # 6. Calculate detailed metrics
    task_metrics = calculate_task_metrics(
        model, val_loader, target_cols, MULTI_LABEL_COLUMN_NAMES, device
    )

    # 7. Calculate normalized loss
    max_loss_list = []
    for i, col in enumerate(target_cols):
        if col in MULTI_LABEL_COLUMN_NAMES:
            max_loss_list.append(math.log(2))
        else:
            n_classes = n_targets_list[i]
            max_loss_list.append(math.log(n_classes) if n_classes > 1 else 0.0)

    max_loss = sum(max_loss_list) if max_loss_list else 1.0
    normalized_loss = best_loss / max_loss if max_loss > 0 else best_loss

    avg_accuracy = (
        sum(best_accuracies_list) / len(best_accuracies_list)
        if best_accuracies_list
        else 0.0
    )

    # 8. Store metrics in trial
    trial.set_user_attr("avg_accuracy", avg_accuracy)
    trial.set_user_attr("normalized_loss", normalized_loss)
    trial.set_user_attr("all_accuracies", best_accuracies_list)
    trial.set_user_attr("seed", RANDOM_SEED)

    # Store per-task metrics
    for col, metrics in task_metrics.items():
        for metric_name, value in metrics.items():
            trial.set_user_attr(f"{col}_{metric_name}", value)

    # 9. Identify critical task metric
    critical_task = "effect_type"
    critical_f1 = task_metrics.get(critical_task, {}).get("f1_weighted", 0.0)

    trial.set_user_attr("critical_task_f1", critical_f1)
    trial.set_user_attr("critical_task_name", critical_task)

    # 10. Update progress bar
    info_dict = {
        "Trial": trial.number,
        "Loss": f"{best_loss:.4f}",
        "NormLoss": f"{normalized_loss:.4f}",
        f"{critical_task}_F1": f"{critical_f1:.4f}",
        "AvgAcc": f"{avg_accuracy:.4f}",
    }

    pbar.set_postfix(info_dict, refresh=True)
    pbar.update(1)

    # 11. Return objectives
    if use_multi_objective:
        # Multi-objective: minimize loss AND maximize F1 (negated for minimization)
        return best_loss, -critical_f1
    else:
        # Single-objective: only loss
        return (best_loss,)


def get_best_trials(
    study: optuna.Study, n: int = 5
) -> List[optuna.trial.FrozenTrial]:
    """
    Get the top N valid trials from a study.
    
    Handles both single and multi-objective studies. For multi-objective,
    returns trials from the Pareto front.
    
    Args:
        study: Optuna study object
        n: Number of trials to return
    
    Returns:
        List of top N trials, sorted by objective values
    """
    valid_trials = [
        t for t in study.trials
        if t.values is not None and 
           all(isinstance(v, (int, float)) and not math.isnan(v) for v in t.values) and
           t.state == optuna.trial.TrialState.COMPLETE
    ]

    if not valid_trials:
        logger.warning("No valid completed trials found in study")
        return []

    # Para multi-objetivo, usar best_trials (Pareto front)
    if len(valid_trials[0].values) > 1:
        try:
            # Obtener trials del frente de Pareto
            pareto_trials = study.best_trials
            # Limitar a N trials
            return pareto_trials[:n] if pareto_trials else valid_trials[:n]
        except Exception as e:
            logger.warning(f"Error getting Pareto trials: {e}")
            # Fallback: ordenar por primera métrica
            return sorted(valid_trials, key=lambda t: t.values[0])[:n]
    else:
        # Para single-objetivo, ordenar por valor
        return sorted(valid_trials, key=lambda t: t.values[0])[:n]


def _save_optuna_report(
    study: optuna.Study,
    model_name: str,
    results_file_json: Path,
    results_file_txt: Path,
    top_5_trials: List[optuna.trial.FrozenTrial],
    is_multi_objective: bool = False,
) -> None:
    """
    Save Optuna optimization results to JSON and TXT files.
    
    Args:
        study: Optuna study object
        model_name: Name of the model
        results_file_json: Path to save JSON report
        results_file_txt: Path to save TXT report
        top_5_trials: List of top trials to include
        is_multi_objective: Whether study was multi-objective
    
    Raises:
        IOError: If files cannot be written
    """
    if not top_5_trials:
        logger.warning("No valid trials found to save in report")
        return

    # Para multi-objetivo, el "mejor" trial es subjetivo, tomar el primero del Pareto front
    best_trial = top_5_trials[0]

    # Save JSON
    output = {
        "model": model_name,
        "optimization_type": "multi_objective" if is_multi_objective else "single_objective",
        "n_trials": len(study.trials),
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial": {
            "number": best_trial.number,
            "values": list(best_trial.values) if best_trial.values else [],
            "params": best_trial.user_attrs.get("full_params", best_trial.params),
            "user_attrs": dict(best_trial.user_attrs),
        },
        "clinical_priorities": CLINICAL_PRIORITIES,
    }
    
    if is_multi_objective:
        # Para multi-objetivo, incluir todos los trials del Pareto front
        output["pareto_front"] = [
            {
                "number": t.number,
                "values": list(t.values) if t.values else [],
                "params": t.user_attrs.get("full_params", t.params),
                "critical_f1": t.user_attrs.get("critical_task_f1", 0.0),
            }
            for t in top_5_trials
        ]
    else:
        # Para single-objetivo, incluir top 5 por loss
        output["top5"] = [
            {
                "number": t.number,
                "values": list(t.values) if t.values else [],
                "params": t.params,
                "critical_f1": t.user_attrs.get("critical_task_f1", 0.0),
            }
            for t in top_5_trials
        ]

    try:
        with open(results_file_json, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"JSON report saved: {results_file_json}")
    except IOError as e:
        logger.error(f"Failed to save JSON report: {e}")
        raise

    # Save TXT
    try:
        with open(results_file_txt, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"OPTIMIZATION REPORT: {model_name}\n")
            f.write("=" * 70 + "\n\n")

            if is_multi_objective:
                f.write("Optimization Type: Multi-Objective\n")
                f.write("   - Objective 1: Minimize Loss\n")
                f.write(f"   - Objective 2: Maximize F1 ({best_trial.user_attrs.get('critical_task_name', 'N/A')})\n\n")
                f.write("Note: Results show Pareto-optimal solutions\n\n")
            else:
                f.write("Optimization Type: Single-Objective (Loss)\n\n")

            f.write(f"Representative Trial (#{best_trial.number})\n")
            f.write("-" * 60 + "\n\n")

            # Objective values
            if best_trial.values:
                f.write("Objective Values:\n")
                if is_multi_objective:
                    f.write(f"   Loss:         {best_trial.values[0]:.5f}\n")
                    f.write(f"   F1 (negated): {best_trial.values[1]:.5f} → F1 actual: {-best_trial.values[1]:.5f}\n")
                else:
                    f.write(f"   Loss:         {best_trial.values[0]:.5f}\n")
                f.write("\n")

            # Detailed metrics
            f.write("Detailed Metrics:\n")
            f.write(f"   Normalized Loss:    {best_trial.user_attrs.get('normalized_loss', 0.0):.5f}\n")
            f.write(f"   Average Accuracy:   {best_trial.user_attrs.get('avg_accuracy', 0.0):.4f}\n")
            f.write(f"   Critical Task F1:   {best_trial.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
            f.write("\n")

            # Hyperparameters
            f.write("Hyperparameters:\n")
            # Use full_params from user_attrs if available, otherwise fallback to trial.params
            report_params = best_trial.user_attrs.get("full_params", best_trial.params)
            for key, value in sorted(report_params.items()):
                f.write(f"   {key:25s}: {value}\n")
            f.write("\n")

            # Top results
            if is_multi_objective:
                f.write("Pareto Front (Top Solutions)\n")
                f.write("-" * 60 + "\n\n")
                for i, t in enumerate(top_5_trials, 1):
                    f.write(f"{i}. Trial #{t.number}\n")
                    if t.values:
                        f.write(f"   Loss: {t.values[0]:.5f}, F1: {-t.values[1]:.5f}\n")
                    f.write(f"   Critical F1: {t.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
                    all_trial_params = t.user_attrs.get("full_params", t.params)
                    f.write(f"   Key params: batch_size={all_trial_params.get('batch_size')}, lr={all_trial_params.get('learning_rate'):.2e}\n\n")
                                        
            else:
                f.write("Top 5 Trials by Loss\n")
                f.write("-" * 60 + "\n\n")
                for i, t in enumerate(top_5_trials, 1):
                    f.write(f"{i}. Trial #{t.number}\n")
                    if t.values:
                        f.write(f"   Loss: {t.values[0]:.5f}\n")
                    f.write(f"   Critical F1: {t.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
                    all_trial_params = t.user_attrs.get("full_params", t.params)
                    f.write(f"   Key params: batch_size={all_trial_params.get('batch_size')}, lr={all_trial_params.get('learning_rate'):.2e}\n\n")
                    
                    
        logger.info(f"TXT report saved: {results_file_txt}")
    except IOError as e:
        logger.error(f"Failed to save TXT report: {e}")
        raise


def run_optuna_with_progress(
    model_name: str,
    n_trials: int = N_TRIALS,
    output_dir: Path = PGEN_MODEL_DIR,
    target_cols: Optional[List[str]] = None,
    use_multi_objective: bool = True,
) -> Tuple[Dict[str, Any], float, Optional[Path], float]:
    """
    Run Optuna hyperparameter optimization with progress tracking.
    
    Executes a complete optimization study with support for single and
    multi-objective optimization. Generates visualizations and reports.
    
    Args:
        model_name: Name of the model to optimize
        n_trials: Number of trials to run
        output_dir: Directory to save results
        target_cols: List of target columns (if None, uses config)
        use_multi_objective: If True, optimize (loss, -F1). If False, only loss.
    
    Returns:
        Tuple of (best_params, best_loss, results_json_path, normalized_loss)
        Returns empty dict and None values if optimization fails.
    
    Raises:
        ValueError: If model_name not found or n_trials invalid
    """
    if n_trials <= 0:
        raise ValueError(f"n_trials must be positive, got {n_trials}")

    datetime_study = datetime.datetime.now().strftime("%d_%m_%y__%H_%M")
    output_dir = Path(output_dir)
    results_dir = output_dir / "optuna_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Define targets
    config = get_model_config(model_name)
    inputs_from_config = [i.lower() for i in config["inputs"]]
    targets_from_config = [t.lower() for t in config["targets"]]
    cols_from_config = config["cols"]

    if target_cols is None:
        target_cols = targets_from_config

    # 2. Load and preprocess data
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION - PHARMAGEN PMODEL")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(
        f"Mode: {'Multi-Objective (Loss + F1)' if use_multi_objective else 'Single-Objective (Loss)'}"
    )
    print(f"Trials: {n_trials}")
    print("=" * 60 + "\n")

    logger.info("Loading and preprocessing data...")
    try:
        csvfiles, _ = train_data_import(targets_from_config)
    except FileNotFoundError as e:
        logger.error(f"No se pudo encontrar el archivo de datos: {e}")
        return {}, float("inf"), None, 0.0

    
    data_loader = PGenDataProcess()
    df = data_loader.load_data(
        csv_path=str(csvfiles),
        all_cols=cols_from_config,
        input_cols=inputs_from_config,
        target_cols=targets_from_config,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=["effect_type"]
    )
    
    train_df, val_df = train_test_split(
        df,
        test_size=VALIDATION_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["stratify_col"],
    )

    data_loader.fit(train_df)
    train_processed_df = data_loader.transform(train_df)
    val_processed_df = data_loader.transform(val_df)
    logger.info("Data ready.")

    # 3. Calculate class weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable GPU optimizations if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("GPU optimizations enabled (cuDNN benchmark, TF32)")
    
    class_weights_task3 = None
    task3_name = "effect_type"

    # Comprueba si la tarea está activa en la configuración actual
    if task3_name in targets_from_config:
        logger.info(f"Calculando pesos de clase para la tarea '{task3_name}'...")
        try:
            encoder_task3 = data_loader.encoders[task3_name]
            class_counts = train_processed_df[task3_name].value_counts()
            num_classes = len(encoder_task3.classes_)
            all_counts = torch.zeros(num_classes)
            
            # Asegurarse de que los índices son válidos
            valid_indices = [idx for idx in class_counts.index if idx < num_classes]
            valid_counts = class_counts[valid_indices]

            all_counts[valid_counts.index] = torch.tensor(valid_counts.values, dtype=torch.float32) # type: ignore
            weights = 1.0 / torch.log(all_counts + CLASS_WEIGHT_LOG_SMOOTHING)
            
            weights[torch.isinf(weights)] = 1.0 # Asigna un peso neutral a clases no vistas en train
            weights[torch.isnan(weights)] = 1.0 # Por si acaso

            weights = weights / weights.mean()
            class_weights_task3 = weights.to(device)

            logger.info(
                f"Pesos de clase calculados para '{task3_name}' ({num_classes} clases): "
                f"Min={weights.min():.3f}, Max={weights.max():.3f}, Mean={weights.mean():.3f}"
            )
        except KeyError:
            logger.warning(
                f"No se pudo calcular los pesos: el encoder para '{task3_name}' no fue encontrado. "
                "Asegúrate de que esté en 'target_cols' y presente en los datos de entrenamiento."
            )
        except Exception as e:
            logger.error(f"Ocurrió un error inesperado al calcular los pesos de clase: {e}")

    # 4. Configure sampler
    sampler = TPESampler(
        n_startup_trials=N_STARTUP_TRIALS,
        n_ei_candidates=N_EI_CANDIDATES,
        multivariate=True,
        seed=RANDOM_SEED,
    )

    # 5. Run optimization
    pbar = tqdm(
        total=n_trials,
        desc=f"Optuna ({model_name})",
        colour="green",
        leave=False,
        unit="trial",
    )

    optuna_func = partial(
        optuna_objective,
        model_name=model_name,
        target_cols=target_cols,
        pbar=pbar,
        fitted_data_loader=data_loader,
        train_processed_df=train_processed_df,
        val_processed_df=val_processed_df,
        csvfiles=csvfiles,
        class_weights_task3=class_weights_task3,
        use_multi_objective=use_multi_objective,
    )

    
    study_name=f"optuna_{model_name}_{datetime_study}"
    
    # Create study based on mode
    if use_multi_objective:
        pruner = None
        logger.info("Pruning disabled (incompatible with multi-objective)")
        study = optuna.create_study(
            directions=["minimize", "minimize"],
            study_name=study_name,
            storage = f"sqlite:///{MODELS_DIR}/{study_name}.db",
            sampler=sampler,
            pruner=pruner,
        )
    else:
        pruner = MedianPruner(
            n_startup_trials=N_PRUNER_STARTUP_TRIALS,
            n_warmup_steps=N_PRUNER_WARMUP_STEPS,
            interval_steps=PRUNER_INTERVAL_STEPS,
        )
        logger.info("Pruning enabled (single-objective)")
        study = optuna.create_study(
            direction="minimize",
            study_name=f"optuna_{model_name}_{datetime_study}",
            storage = f"sqlite:///{MODELS_DIR}/{study_name}.db", # Guardar también el estudio single-objective
            sampler=sampler,
            pruner=pruner,
        )

    study.optimize(optuna_func, n_trials=n_trials, gc_after_trial=True)
    
    
    pbar.close()

    # 6. Save visualizations
    if study.trials:
        try:
            # Optimization history
            fig_history = plot_optimization_history(study)
            graph_filename = optuna_results / f"history_{model_name}_{datetime_study}.html"
            fig_history.write_html(str(graph_filename))
            logger.info(f"Optimization history saved: {graph_filename}")

            # Pareto front (multi-objective only)
            if use_multi_objective:
                # Para multi-objetivo, usamos study.best_trials en lugar de study.best_trial
                if len(study.best_trials) > 1:
                    try:
                        fig_pareto = plot_pareto_front(study)
                        pareto_filename = optuna_results / f"pareto_{model_name}_{datetime_study}.html"
                        fig_pareto.write_html(str(pareto_filename))
                        logger.info(f"Pareto front saved: {pareto_filename}")
                    except Exception as pareto_error:
                        logger.warning(f"Error creating Pareto front plot: {pareto_error}")
                else:
                    logger.info("Not enough trials for Pareto front visualization (need at least 2)")
            
            # Visualizaciones adicionales para ambos modos
            try:
                # Parameter importances (solo funciona con single-objective)
                if not use_multi_objective and len(study.trials) >= 10:
                    from optuna.visualization import plot_param_importances
                    fig_importance = plot_param_importances(study)
                    importance_filename = optuna_results / f"param_importance_{model_name}_{datetime_study}.html"
                    fig_importance.write_html(str(importance_filename))
                    logger.info(f"Parameter importance saved: {importance_filename}")
            except Exception as importance_error:
                logger.debug(f"Could not create parameter importance plot: {importance_error}")
                
            try:
                # Slice plot (funciona para ambos modos)
                from optuna.visualization import plot_slice
                fig_slice = plot_slice(study)
                slice_filename = optuna_results / f"slice_{model_name}_{datetime_study}.html"
                fig_slice.write_html(str(slice_filename))
                logger.info(f"Slice plot saved: {slice_filename}")
            except Exception as slice_error:
                logger.debug(f"Could not create slice plot: {slice_error}")

        except Exception as e:
            logger.warning(f"Error saving visualizations: {e}")

    # 7. Get and save report
    best_5 = get_best_trials(study, 5)

    if not best_5:
        logger.error("Optimization completed but no valid trials found")
        return {}, float("inf"), None, 0.0

    filename = f"optuna_{model_name}_{datetime_study}"
    results_file_json = results_dir / f"{filename}.json"
    results_file_txt = results_dir / f"{filename}.txt"

    _save_optuna_report(
        study,
        model_name,
        results_file_json,
        results_file_txt,
        best_5,
        is_multi_objective=use_multi_objective,
    )

    # Extract metrics from best trial
    best_trial = best_5[0]
    best_params = best_trial.user_attrs.get("full_params", best_trial.params)
    best_loss = best_trial.values[0] if best_trial.values else float("inf")
    normalized_loss = best_trial.user_attrs.get("normalized_loss", 0.0)
    best_f1 = best_trial.user_attrs.get("critical_task_f1", 0.0)

    # Print summary
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETED")
    print("=" * 70)
    
    if use_multi_objective:
        print(f"Mode: Multi-Objective Optimization")
        print(f"Pareto Front Solutions: {len(study.best_trials)}")
        print(f"\nRepresentative Solution (Trial #{best_trial.number}):")
        print(f"   Loss:           {best_loss:.5f}")
        print(f"   F1 (Critical):  {best_f1:.5f}")
        if len(best_trial.values) > 1:
            print(f"   F1 (Objective): {-best_trial.values[1]:.5f}")
    else:
        print(f"Mode: Single-Objective Optimization")
        print(f"Best Trial: #{best_trial.number}")
        print(f"   Loss:           {best_loss:.5f}")
        print(f"   F1 (Critical):  {best_f1:.5f}")

    print(f"\nCompleted Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/{len(study.trials)}")
    print(f"Reports saved in: {results_dir}")
    print("=" * 70 + "\n")

    return best_params, best_loss, results_file_json, normalized_loss


# ============================================================================
# 9. PUNTO DE ENTRADA (Ejemplo)
# ============================================================================

if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Este es un punto de entrada de ejemplo.
    # Necesitarás tus archivos 'model_configs.py' y 'focal_loss.py'
    # en el mismo directorio, así como tu estructura de datos (ej. ./data)
    # y la configuración de 'src.config.config'.
    # ----------------------------------------------------------------------
    
    logger.info("Ejecutando el script consolidado...")
    
    # --- EJEMPLO 1: Ejecutar el pipeline de entrenamiento final ---
    # (Asegúrate de que 'My_Model_Name' existe en tu 'model_configs.py')
    
    MODEL_NAME_TO_TRAIN = "Phenotype_Effect_Outcome_Kv" # Ej: "Phenotype_Effect_Outcome"
    PGEN_MODEL_DIR_PATH = "./models/pgen" # Directorio de salida
    
    try:
        logger.info(f"Iniciando train_pipeline para '{MODEL_NAME_TO_TRAIN}'...")
        train_pipeline(
            PGEN_MODEL_DIR_STR=PGEN_MODEL_DIR_PATH,
            csv_files=None, # 'csv_files' es obsoleto, se carga desde config
            model_name=MODEL_NAME_TO_TRAIN,
            target_cols=None, # Usará los targets de la config
            patience=PATIENCE,
            epochs=EPOCHS
        )
        logger.info("train_pipeline completado.")
        
    except Exception as e:
        logger.error(f"Falló la ejecución de train_pipeline: {e}")
        import traceback
        traceback.print_exc()

    '''
    # --- EJEMPLO 2: Ejecutar la optimización de Optuna ---
    # (Asegúrate de que 'My_Model_Name' existe en tu 'model_configs.py')
    
    MODEL_NAME_TO_OPTIMIZE = "Phenotype_Effect_Outcome_Kv"
    N_TRIALS_OPTUNA = 40 # Un número bajo para probar
    
    try:
        logger.info(f"Iniciando run_optuna_with_progress para '{MODEL_NAME_TO_OPTIMIZE}'...")
        run_optuna_with_progress(
             model_name=MODEL_NAME_TO_OPTIMIZE,
             n_trials=N_TRIALS_OPTUNA,
             output_dir=Path(PGEN_MODEL_DIR),
             target_cols=None, # Usará los targets de la config
             use_multi_objective=True # O False para single-objective
         )
        logger.info("run_optuna_with_progress completado.")
        
    except Exception as e:
         logger.error(f"Falló la ejecución de run_optuna_with_progress: {e}")
         import traceback
         traceback.print_exc()

    logger.warning("Fin del script. Descomenta el bloque `if __name__ == '__main__':` para ejecutar un pipeline.")
    '''