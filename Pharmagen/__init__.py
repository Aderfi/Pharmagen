# Pharmagen/__init__.py

if __name__ == "Pharmagen":
    import sys
    from pathlib import Path

    PHARMAGEN_DIR = Path(__file__).resolve()
    if str(PHARMAGEN_DIR) not in sys.path:
        sys.path.insert(0, str(PHARMAGEN_DIR))

from . import *

from .config import (
    DATA_DIR,
    LOGS_DIR,
    CACHE_DIR,
    SRC_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    DOCS_DIR,
    SCRIPTS_DIR,
    ANALYSIS_DIR,
    DATA_HANDLER_DIR,
    ENV_SCRIPTS_DIR,
    DL_MODEL_DIR,
    MODEL_SCRIPTS_DIR,
    MODELS_DIR,
    MODEL_VOCABS_DIR,
    MODEL_DATA_DIR,
    MODEL_CSV_DIR,
    MODEL_JSON_DIR,
    CONFIG_FILE,
    MODEL_TRAIN_DATA,
    MODELS_DIR,
)

__all__ = [
    "DATA_DIR",
    "LOGS_DIR",
    "CACHE_DIR",
    "SRC_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "RESULTS_DIR",
    "DOCS_DIR",
    "SCRIPTS_DIR",
    "ANALYSIS_DIR",
    "DATA_HANDLER_DIR",
    "ENV_SCRIPTS_DIR",
    "DL_MODEL_DIR",
    "MODEL_SCRIPTS_DIR",
    "MODELS_DIR",
    "MODEL_VOCABS_DIR",
    "MODEL_DATA_DIR",
    "MODEL_CSV_DIR",
    "MODEL_JSON_DIR",
    "CONFIG_FILE",
    "MODEL_TRAIN_DATA",
    "MODELS_DIR",
]