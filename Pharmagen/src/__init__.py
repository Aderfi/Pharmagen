"""
Módulo de Análisis Bioinformático


Este paquete contiene todas las funciones relacionadas con el análisis
de secuencias, como la búsqueda de mutaciones, alineamos y extracción
de características genómicas relevantes.

"""

from . import utils, data_handle, analysis, funciones, logger_config, project_config, config
from config import *

__utils__ = ["utils"]
__data_handle__ = ["io", "processing"]
__funciones__ = ["mutations_extraction", "genoma_humano_process"]
__analysis__ = ["modeling", "visualization"]
__config__= [PHARMAGEN_DIR, LOGS_DIR, CACHE_DIR, SRC_DIR, \
        RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, DOCS_DIR, SCRIPTS_DIR, ENV_SCRIPTS_DIR, \
        DL_MODEL_DIR, MODEL_SCRIPTS_DIR, MODELS_DIR, MODEL_VOCABS_DIR, MODEL_DATA_DIR, MODEL_CSV_DIR, MODEL_JSON_DIR, \
        CONFIG_FILE, LOG_FILE, AUTOR, VERSION, FECHA_CREACION \
            ]