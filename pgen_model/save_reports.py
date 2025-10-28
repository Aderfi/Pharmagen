import json
import os
from src.config.config import *
import datetime
import pandas as pd 


def type_check(report_data):
    return type(report_data)

 # Formatos soportados
def save_report(data, model_name, output_as: str, report_name: str, target_cols: list, save_dir: str):
    """
    Guarda los reportes de entrenamiento en el formato especificado.

    Args:
        data (dict/list/str): Datos del reporte a guardar.
        output_as (str): Extensión del archivo de salida. Opciones: ".json", ".csv", ".txt". (Nada)""
        model_name (str): Indica la carpeta en la que se guardará el reporte.
        report_name (str): Nombre base del archivo (sin extensión).

    Returns:
        str: Ruta completa del archivo guardado.
        report_name (str): Nombre base del archivo (sin extensión).
        save_dir (str/Path): Directorio donde guardar el reporte.
        save_as (str): Formato de guardado. Soportados: ".json", ".csv", ".txt".

    Returns:
        str: Ruta completa del archivo guardado.
    """
    
    
    

    return