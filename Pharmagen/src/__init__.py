"""
Módulo de Análisis Bioinformático


Este paquete contiene todas las funciones relacionadas con el análisis
de secuencias, como la búsqueda de mutaciones, alineamos y extracción
de características genómicas relevantes.

"""

from . import utils, data_handle, analysis, funciones, logger_config, project_config

__all__ = ["__analysis__", "__data_handle__", "__utils__", "__funciones__", "logger_config", "project_config"]
__utils__ = ["utils"]
__data_handle__ = ["io", "processing"]
__funciones__ = ["mutations_extraction", "genoma_humano_process"]
__analysis__ = ["modeling", "visualization"]