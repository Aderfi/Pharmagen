from pathlib import Path
from datetime import datetime
import os, sys

# === RUTAS BASE ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Pharmagen
# === FECHA PARA LOGS Y OTROS ===
DATE_STAMP = datetime.now().strftime("%Y%m%d")

# === RUTAS DE CARPETAS PRINCIPALES PHARMAGEN ===
CACHE_DIR = PROJECT_ROOT / "cache"             # Carpeta para datos temporales
DATA_DIR = PROJECT_ROOT / "data"               # Carpeta para datos
LOGS_DIR = PROJECT_ROOT / "logs"               # Carpeta para logs
SRC_DIR = PROJECT_ROOT / "src"                 # Carpeta para código fuente
SCRIPTS_DIR = PROJECT_ROOT / "scripts"         # Carpeta para scripts
ENV_SCRIPTS_DIR = PROJECT_ROOT / "venv_utils"  # Carpeta para scripts de entorno virtual
RESULTS_DIR = PROJECT_ROOT / "results"         # Carpeta para resultados

# === RUTAS DE MODELOS DEEP LEARNING ===
PGEN_MODEL_DIR = PROJECT_ROOT / "pgen_model"    # Carpeta principal del modelo
MODEL_SCRIPTS_DIR = PGEN_MODEL_DIR / "scripts"   # Carpeta para scripts del modelo
MODELS_DIR = PGEN_MODEL_DIR / "models"           # Carpeta para modelos entrenados
MODEL_LABEL_VOCABS_DIR = PGEN_MODEL_DIR / "labels_vocabs"     # Carpeta para vocabularios y tokenizadores
MODEL_DATA_DIR = PGEN_MODEL_DIR / "docs_data"    # Carpeta para datos del modelo
MODEL_CSV_DIR = MODEL_DATA_DIR / "csv"           # Carpeta para datos CSV del modelo

# Dentro de MODEL_DATA_DIR, subcarpetas específicas
MODEL_JSON_DIR = MODEL_DATA_DIR / "json"         # Carpeta para datos JSON del modelo
MODEL_TRAIN_DATA = MODEL_DATA_DIR / "train_csv"  # Carpeta para datos de entrenamiento del modelo
MODEL_FILES_TRAINING = MODEL_TRAIN_DATA / "files" # Carpeta para archivos de entrenamiento del modelo

# === RUTAS DE ARCHIVOS ESPECÍFICOS ===
CONFIG_FILE = PROJECT_ROOT / "config.json"
LOG_FILE = LOGS_DIR / f"pharmagen_{DATE_STAMP}.log"

# === METADATOS DEL SOFTWARE ===
AUTOR = "Astordna/Aderfi/Adrim Hamed Outmani"
VERSION = "0.1"
FECHA_CREACION = "2024-06-15"

def __dirPaths__():
    return {
        "Project_root": PROJECT_ROOT,
        "Pharmagen": PROJECT_ROOT,
        "CACHE_DIR": CACHE_DIR,
        "DATA_DIR": DATA_DIR,
        "LOGS_DIR": LOGS_DIR,
        "SRC_DIR": SRC_DIR,
        "SCRIPTS_DIR": SCRIPTS_DIR,
        "ENV_SCRIPTS_DIR": ENV_SCRIPTS_DIR,
        "RESULTS_DIR": RESULTS_DIR,
        "PGEN_MODEL_DIR": PGEN_MODEL_DIR,
        "MODEL_SCRIPTS_DIR": MODEL_SCRIPTS_DIR,
        "MODELS_DIR": MODELS_DIR,
        "MODEL_VOCABS_DIR": MODEL_VOCABS_DIR,
        "MODEL_DATA_DIR": MODEL_DATA_DIR,
        "MODEL_CSV_DIR": MODEL_CSV_DIR,
        "MODEL_JSON_DIR": MODEL_JSON_DIR,
        "MODEL_TRAIN_DATA": MODEL_TRAIN_DATA,
        "MODEL_FILES_TRAINING": MODEL_FILES_TRAINING,
        "CONFIG_FILE": CONFIG_FILE,
        "LOG_FILE": LOG_FILE
    }