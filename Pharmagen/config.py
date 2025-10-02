from pathlib import Path
from datetime import datetime

date = datetime.now().strftime("%Y%m%d")


### !!!!!!! IMPORTANTE !!!!!!! ###
##    CAMBIAR SOLO ESTA RUTA    ##
PHARMAGEN_DIR = Path(__file__).resolve().parent
##################################

# --- Rutas del Proyecto ---
LOGS_DIR = PHARMAGEN_DIR / "logs"
CACHE_DIR = PHARMAGEN_DIR / "cache" 
SRC_DIR = PHARMAGEN_DIR / "src"
RAW_DATA_DIR = PHARMAGEN_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PHARMAGEN_DIR / "data" / "processed"
RESULTS_DIR = PHARMAGEN_DIR / "results"
DOCS_DIR = PHARMAGEN_DIR / "docs"
SCRIPTS_DIR = PHARMAGEN_DIR / "scripts"
ENV_SCRIPTS_DIR = PHARMAGEN_DIR.parent / "Environment_Scripts"

# --- Rutas del Modelo --- DEEPL_MODEL_DIR

DL_MODEL_DIR = PHARMAGEN_DIR / "deepL_model"
MODEL_SCRIPTS_DIR = DL_MODEL_DIR / "scripts"
MODELS_DIR = DL_MODEL_DIR / "models"
MODEL_VOCABS_DIR = DL_MODEL_DIR / "vocabs"
MODEL_DATA_DIR = DL_MODEL_DIR / "data"
MODEL_CSV_DIR = MODEL_DATA_DIR / "csv"
MODEL_JSON_DIR = MODEL_DATA_DIR / "json"


# --- Rutas de Archivos Específicos ---

CONFIG_FILE = PHARMAGEN_DIR / "config.json"
LOG_FILE = LOGS_DIR / f"pharmagen_{date}.log"


# --- Metadatos del Software ---

AUTOR = "Astordna/Aderfi/Adrim Hamed Outmani"
VERSION = "0.1"
FECHA_CREACION = "2024-06-15"


