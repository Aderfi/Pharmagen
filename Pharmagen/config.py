from pathlib import Path
from datetime import datetime

# === RUTAS BASE ===
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
PHARMAGEN_DIR = PROJECT_ROOT / "Pharmagen"

# === FECHA PARA LOGS Y OTROS ===
DATE_STAMP = datetime.now().strftime("%Y%m%d")

# === RUTAS DE CARPETAS PRINCIPALES ===
DATA_DIR = PHARMAGEN_DIR / "data"
LOGS_DIR = PHARMAGEN_DIR / "logs"
CACHE_DIR = PHARMAGEN_DIR / "cache"
SRC_DIR = PHARMAGEN_DIR / "src"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PHARMAGEN_DIR / "results"
DOCS_DIR = PHARMAGEN_DIR / "docs"
SCRIPTS_DIR = PHARMAGEN_DIR / "scripts"
ANALYSIS_DIR = PHARMAGEN_DIR / "analysis"
DATA_HANDLER_DIR = SRC_DIR / "data_handle"
ENV_SCRIPTS_DIR = PROJECT_ROOT / "Environment_Scripts"

# === RUTAS DE MODELOS DEEP LEARNING ===
DL_MODEL_DIR = PHARMAGEN_DIR / "deepL_model"
MODEL_SCRIPTS_DIR = DL_MODEL_DIR / "scripts"
MODELS_DIR = DL_MODEL_DIR / "models"
MODEL_VOCABS_DIR = DL_MODEL_DIR / "vocabs"
MODEL_DATA_DIR = DL_MODEL_DIR / "docs_data"
MODEL_CSV_DIR = MODEL_DATA_DIR / "csv"
MODEL_JSON_DIR = MODEL_DATA_DIR / "json"
MODEL_TRAIN_DATA = MODEL_DATA_DIR / "train_csv"
MODEL_FILES_TRAINING = MODEL_TRAIN_DATA / "files"

# === RUTAS DE ARCHIVOS ESPECÍFICOS ===
CONFIG_FILE = PHARMAGEN_DIR / "config.json"
LOG_FILE = LOGS_DIR / f"pharmagen_{DATE_STAMP}.log"

# === METADATOS DEL SOFTWARE ===
AUTOR = "Astordna/Aderfi/Adrim Hamed Outmani"
VERSION = "0.1"
FECHA_CREACION = "2024-06-15"