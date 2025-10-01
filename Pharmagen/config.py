from pathlib import Path
from datetime import datetime
date = datetime.now().strftime("%Y%m%d")

# --- Rutas del Proyecto ---

PROJECT_ROOT = Path(__file__).resolve().parent
PHARMAGEN_DIR = PROJECT_ROOT / "Pharmagen"
LOGS_DIR = PHARMAGEN_DIR / "logs"
CACHE_DIR = PHARMAGEN_DIR / "cache" # Renombrado para claridad
SRC_DIR = PHARMAGEN_DIR / "src"
RAW_DATA_DIR =  PHARMAGEN_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PHARMAGEN_DIR / "data" / "processed"
RESULTS_DIR = PHARMAGEN_DIR / "results"
DOCS_DIR = PHARMAGEN_DIR / "docs"
SCRIPTS_DIR = PHARMAGEN_DIR / "scripts"
ENV_SCRIPTS_DIR = PHARMAGEN_DIR / "Environment_Scripts"

# --- Rutas de Archivos Específicos ---

CONFIG_FILE = PHARMAGEN_DIR / "config.json"
LOG_FILE = LOGS_DIR / f"pharmagen_{date}.log"


# --- Metadatos del Software ---

AUTOR = "Astordna/Aderfi/Adrim Hamed Outmani"
VERSION = "0.1"
FECHA_CREACION = "2024-06-15"