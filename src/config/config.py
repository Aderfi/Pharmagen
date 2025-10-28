from pathlib import Path
from datetime import datetime
import os, sys

# === RUTAS BASE ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Pharmagen
# === FECHA PARA LOGS Y OTROS ===
DATE_STAMP = datetime.now().strftime("%Y%m%d")

"""
------ DIRECTORIOS DEL PROYECTO ------
"""

# === RUTAS DE CARPETAS PRINCIPALES PHARMAGEN ===
CACHE_DIR = PROJECT_ROOT / "cache"  # Carpeta para datos temporales
DATA_DIR = PROJECT_ROOT / "data"  # Carpeta para datos
LOGS_DIR = PROJECT_ROOT / "logs"  # Carpeta para logs
SRC_DIR = PROJECT_ROOT / "src"  # Carpeta para código fuente
ENV_SCRIPTS_DIR = PROJECT_ROOT / "venv_utils"  # Carpeta para scripts de entorno virtual
RESULTS_DIR = PROJECT_ROOT / "results"

# === SUBCARPETAS DENTRO DE SRC ===
SCRIPTS_DIR = SRC_DIR / "scripts"  # Carpeta para scripts
DATA_HANDLE_DIR = SRC_DIR / "data_handle"  # Carpeta para manejo de datos
CONFIG_DIR = SRC_DIR / "config"  # Carpeta para configuración
ANALYSIS_DIR = SRC_DIR / "analysis"  # Carpeta para análisis

# === SUBCARPETAS DENTRO DE DATA ===
RAW_DATA_DIR = DATA_DIR / "raw"  # Carpeta para datos sin procesar
GENE_RAW_DATA_DIR = (
    RAW_DATA_DIR / "genotype_raw"
)  # Carpeta para datos genéticos sin procesar
DRUG_RAW_DATA_DIR = (
    RAW_DATA_DIR / "drug_raw"
)  # Carpeta para datos de fármacos sin procesar


PROCESSED_DATA_DIR = DATA_DIR / "processed"  # Carpeta para datos procesados


"""
------ DIRECTORIOS DEL MODELO DE DEEP LEARNING ------
        
"""
# === RUTAS DE MODELOS DEEP LEARNING ===
PGEN_MODEL_DIR = PROJECT_ROOT / "pgen_model"  # Carpeta principal del modelo
# --------------------------------------------------------------------------------
MODEL_SCRIPTS_DIR = PGEN_MODEL_DIR / "scripts"  # Carpeta para scripts del modelo
MODELS_DIR = PGEN_MODEL_DIR / "models"  # Carpeta para modelos entrenados
MODEL_LABEL_VOCABS_DIR = (
    PGEN_MODEL_DIR / "labels_vocabs"
)  # Carpeta para vocabularios y tokenizadores
MODEL_DATA_DIR = PGEN_MODEL_DIR / "docs_data"  # Carpeta para datos del modelo
MODEL_CSV_DIR = MODEL_DATA_DIR / "csv"  # Carpeta para datos CSV del modelo
MODEL_TRAIN_DATA = (
    PGEN_MODEL_DIR / "train_data"
)  # Carpeta para datos de entrenamiento del modelo
MODEL_ENCODERS_DIR = PGEN_MODEL_DIR / "encoders"  # Carpeta para encoders del modelo


# Dentro de MODEL_DATA_DIR, subcarpetas específicas
MODEL_JSON_DIR = MODEL_DATA_DIR / "json"  # Carpeta para datos JSON del modelo
MODEL_FILES_TRAINING = (
    MODEL_TRAIN_DATA / "files"
)  # Carpeta para archivos de entrenamiento del modelo

# === RUTAS DE ARCHIVOS ESPECÍFICOS ===
CONFIG_FILE = PROJECT_ROOT / "config.json"
LOG_FILE = LOGS_DIR / f"pharmagen_{DATE_STAMP}.log"

# === METADATOS DEL SOFTWARE ===
AUTOR = "Astordna/Aderfi/Adrim Hamed Outmani"
VERSION = "0.1"
FECHA_CREACION = "2024-06-15"


__all__ = [
    "PROJECT_ROOT",
    "DATE_STAMP",
    "CACHE_DIR",
    "DATA_DIR",
    "LOGS_DIR",
    "SRC_DIR",
    "SCRIPTS_DIR",
    "ENV_SCRIPTS_DIR",
    "RESULTS_DIR",
    "PGEN_MODEL_DIR",
    "MODEL_SCRIPTS_DIR",
    "MODELS_DIR",
    "MODEL_LABEL_VOCABS_DIR",
    "MODEL_DATA_DIR",
    "MODEL_CSV_DIR",
    "MODEL_JSON_DIR",
    "MODEL_TRAIN_DATA",
    "MODEL_FILES_TRAINING",
    "CONFIG_FILE",
    "LOG_FILE",
    "AUTOR",
    "VERSION",
    "FECHA_CREACION",
]
