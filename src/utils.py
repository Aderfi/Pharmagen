import json
import sys
import os
import shutil
import logging
import re
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from ..config.config import *
from src.config.config import DATA_DIR
from .data import PGenDataProcess
from .model_configs import MULTI_LABEL_COLUMN_NAMES
from .focal_loss import create_focal_loss


import glob
from pathlib import Path

__all__ = ["mensaje_introduccion", "load_config", "check_config"]


def mensaje_introduccion():
    introduccion = f""""
    ============================================
            pharmagen_pmodel {VERSION}
    ============================================
    Autor: {AUTOR}
    Fecha: {FECHA_CREACION}
    Descripción: Software para predecir la eficacia terapéutica y toxicidades en base a datos genómicos
                 y el entrenamiento de un modelo predictivo de machine learning.
                 
    ============================================

    \t\t\t**ADVERTENCIA IMPORTANTE**
    
    Para asegurar el correcto funcionamiento del software y evitar errores,
    es preciso ejecutar primero el archivo "Create_CONDA_ENV.py" o 
    "CREATE_VENV.py" ubicado en la carpeta Environment_Scripts.
    SOLO SI ES LA PRIMERA VEZ QUE EJECUTAS EL SOFTWARE.
    
    
    Esto creará el entorno virtual de trabajo con las herramientas y librerías necesarias.
    ============================================
    
    Todos los errores y logs se almacenarán en:
    \t\t\t\t{LOGS_DIR}
    
    """
    return introduccion


def load_config():
    json_file = CONFIG_FILE

    if json_file.exists():
        print(f"\n\t\t>>>Cargando historial desde {json_file}...")
        config_df = json.load(open(json_file, "r"))
    else:
        print(
            f"\t\tNo se encontró el archivo de historial en {json_file}. Se creará con las estructuras \
              por defecto."
        )
        config_df = {
            "_comentario": [
                "Almacenamiento de variables globales, funciones y scripts",
                "que el software ya ha utilizado. Por ejemplo, los que configuran",
                "la estructura de directorios, o los que crean los entornos virtuales.",
            ],
            "environment_created": bool(0),
            "dirs_created": bool(0),
            "libs_installed": bool(0),
            "date_database": "",
            "version": VERSION,
            "os": f"{os.name}",  # "NT (Windows)" o "Posix (Linux)"
        }

    return config_df


def check_config(config_df, choice):
    if config_df["environment_created"] is False:
        print(
            f"\n⚠️  El entorno virtual no ha sido creado. \n Se va a ejecutar el script \
            para crearlo."
        )

        try:
            if [choice == "1"] and (sys.platform == "win32"):
                print("\nEjecutando Create_CONDA_ENV.py...")
                os.system(f'python "{ENV_SCRIPTS_DIR}/Create_CONDA_ENV.py"')

            elif [choice == "1"] and (
                sys.platform != "linux"
            ):  # Ejecutar Create_CONDA_ENV.py en Linux/Mac
                print("\nEjecutando Create_CONDA_ENV.py...")
                (f'python "{ENV_SCRIPTS_DIR}/Create_CONDA_ENV.py"')

            elif [choice == "2"]:
                print("\nEjecutando CREATE_VENV.py...")
                os.system(f'python "{ENV_SCRIPTS_DIR}/CREATE_VENV.py"')

        except Exception as e:
            print(f"Error al intentar crear el entorno virtual: {e}")
            return
    return

def _search_files(path_to_search: Path, ext: None | str):
    # Lógica para buscar archivos en el directorio de predicción
    if ext is None:
        files = [i for i in glob.glob(str(path_to_search / '*'))]
    elif ext is not None and isinstance(ext, str):
        files = [i for i in glob.glob(str(path_to_search / f'*.{ext}'))]
    return files

#####################################
#####   Data Utils/Utilities    #####
#####################################

logger = logging.getLogger(__name__)
UNKNOWN_TOKEN = "__UNKNOWN__"


def serialize_multi_labelcols(label_input: Any) -> str:
    """
    Serializa entradas (listas, arrays, strings) a un string ordenado separado por pipes.
    """
    if pd.isna(label_input) or label_input in ["nan", "", "null", None]:
        return ""

    parts: Set[str] = set()

    if isinstance(label_input, str):
        # Divide por comas, puntos y coma o pipes existentes
        parts = {s.strip() for s in re.split(r"[,;|]+", label_input) if s.strip()}
    elif isinstance(label_input, (list, tuple, np.ndarray, set)):
        parts = {str(x).strip() for x in label_input if x is not None}

    return "|".join(sorted(parts))


def drugs_to_atc(
    df: pd.DataFrame, drug_col: str, atc_col: str = "atc", lang_idx: int = 1
) -> pd.DataFrame:
    """
    Mapea nombres de fármacos a códigos ATC con caché y búsqueda fuzzy optimizada.
    """
    json_path = Path(DATA_DIR) / "dicts" / "ATC_drug_med.json"

    if not json_path.exists():
        logger.warning(f"Diccionario ATC no encontrado en {json_path}. Saltando mapeo.")
        df[atc_col] = "No_ATC"
        return df

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            atc_dict = json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo JSON ATC: {e}")
        return df

    # Mapa inverso optimizado (Nombre -> Código)
    atc_dict_rev = {v[lang_idx].lower(): k for k, v in atc_dict.items()}
    atc_keys = list(atc_dict_rev.keys())

    # Cache local para evitar re-calcular fuzzy matches repetidos
    memoization = {}

    def get_atc_codes(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "No_ATC"

        text_lower = text.lower().strip()
        if text_lower in memoization:
            return memoization[text_lower]

        tokens = [p.strip() for p in re.split(r"[;,/|]+", text_lower) if p.strip()]
        codes = []

        for token in tokens:
            if token in atc_dict_rev:
                codes.append(atc_dict_rev[token])
            else:
                # Fuzzy match estricto (score > 90)
                result = process.extractOne(
                    token, atc_keys, scorer=fuzz.WRatio, score_cutoff=90
                )
                if result:
                    match_name = result[0]
                    codes.append(atc_dict_rev[match_name])
                else:
                    codes.append("No_ATC")

        final_code = "|".join(sorted(set(codes))) if codes else "No_ATC"
        memoization[text_lower] = final_code
        return final_code

    logger.info("Mapeando fármacos a ATC...")
    # map() es más rápido que apply() para operaciones elemento a elemento
    df[atc_col] = df[drug_col].map(get_atc_codes).astype("string")
    return df


def data_import_normalize(
    df: pd.DataFrame,
    all_cols: List[str],
    target_cols: List[str],
    multi_label_targets: Optional[List[str]],
    stratify_cols: Union[List[str], str],
) -> pd.DataFrame:

    multi_label_targets = multi_label_targets or []

    # 1. Limpieza Vectorizada de Single Label y Features
    single_label_cols = [
        c for c in all_cols if c in df.columns and c not in multi_label_targets
    ]

    if single_label_cols:
        df[single_label_cols] = df[single_label_cols].fillna("UNKNOWN").astype(str)
        for col in single_label_cols:
            # Reemplazar separadores y espacios
            df[col] = (
                df[col]
                .str.replace(r"[,;]+", "|", regex=True)
                .str.replace(" ", "_", regex=False)
                .str.strip()
                .str.lower()
            )

    # 2. Limpieza Multi Label
    for col in multi_label_targets:
        if col in df.columns:
            df[col] = df[col].apply(serialize_multi_labelcols).str.lower()

    # 3. Creación de columna de estratificación
    s_cols = [stratify_cols] if isinstance(stratify_cols, str) else stratify_cols
    existing_strat_cols = [c for c in s_cols if c in df.columns]

    if existing_strat_cols:
        df["stratify_col"] = df[existing_strat_cols].astype(str).agg("|".join, axis=1)
    else:
        df["stratify_col"] = "default"

    # Normalizar nombres de columnas (minúsculas y sin espacios)
    df.columns = df.columns.str.lower().str.strip()

    return df


######################################
#####   Train Utils/Utilities    #####
######################################

import logging
from typing import Dict, List, Union, Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
from .data import PGenDataProcess
from .model_configs import MULTI_LABEL_COLUMN_NAMES
from .focal_loss import create_focal_loss


logger = logging.getLogger(__name__)


def get_input_dims(data_loader: PGenDataProcess) -> Dict[str, int]:
    """Devuelve el tamaño del vocabulario para cada columna de entrada."""
    dims = {}
    for col, encoder in data_loader.encoders.items():
        # Solo nos importan features, no targets (aunque estén en data_loader)
        if hasattr(encoder, "classes_"):
            dims[col] = len(encoder.classes_)
        else:  # MultiLabel
            dims[col] = len(encoder.classes_)
    return dims


def get_output_sizes(data_loader: PGenDataProcess, target_cols: List[str]) -> List[int]:
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
                raise AttributeError(
                    f"Encoder for '{col}' missing 'classes_' attribute"
                )

            sizes.append(len(encoder.classes_))
        except (KeyError, AttributeError) as e:
            logger.error(f"Error getting output size for '{col}': {e}")
            raise

    return sizes


def calculate_task_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    feature_cols: List[str],
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
            features = {
                col: batch[col].to(device, non_blocking=True) for col in feature_cols
            }
            targets = {
                col: batch[col].to(device, non_blocking=True) for col in target_cols
            }

            # Forward pass
            outputs = model(**features)

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
        preds = torch.cat(all_preds[col]).numpy()
        targets = torch.cat(all_targets[col]).numpy()

        if col in multi_label_cols:
            # Multi-label: use 'samples' average
            metrics[col] = {
                "f1_samples": f1_score(
                    targets, preds, average="samples", zero_division=0
                ),
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
                "f1_weighted": f1_score(
                    targets, preds, average="weighted", zero_division=0
                ),
                "precision_macro": precision_score(
                    targets, preds, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    targets, preds, average="macro", zero_division=0
                ),
            }

    return metrics


def create_optimizer(model: nn.Module, params: Dict[str, Any]) -> torch.optim.Optimizer:
    lr = params.get("learning_rate", 1e-3)
    wd = params.get("weight_decay", 1e-4)
    opt_type = params.get("optimizer_type", "adamw").lower()

    if opt_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def create_scheduler(optimizer: torch.optim.Optimizer, params: Dict[str, Any]):
    stype = params.get("scheduler_type", "plateau")
    if stype == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params.get("scheduler_factor", 0.5),
            patience=params.get("scheduler_patience", 5),
        )
    return None


def create_criterions(
    target_cols: List[str], params: Dict[str, Any], device: torch.device
) -> List[nn.Module]:
    """Crea una lista de funciones de pérdida correspondiente al orden de target_cols."""
    criterions = []
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        elif col == "effect_type":
            # Ejemplo de uso de Focal Loss para una columna específica
            criterions.append(create_focal_loss(gamma=params.get("focal_gamma", 2.0)))
        else:
            criterions.append(
                nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.1))
            )
    return criterions
