# Software: pharmagen_pmodel
# Versión: 0.1
# Autor: Astordna / Aderfi / Adrim Hamed Outmani
# Fecha: 2024-06-15
# Descripción: Este software tiene la utilidad de actuar como puente para los inputs y la interpretación
#              de los outputs de un modelo predictivo cuya finalidad es inferir a partir de datos genómicos
#              del paciente, la eficacia terapéutica y el posible riesgo incrementado de ciertas toxicidades.
#              El modelo predictivo está basado en un conjunto de modelos de machine learning entrenados
#              con datos genómicos y clínicos de pacientes reales.

import json, os, sys, src  # Asegura que el directorio src es tratado como un paquete
from pathlib import Path
from src.logger_config import unit_logging
from Pharmagen.config import *

unit_logging()

# --- 1. Configuración de Rutas e Importación y About---
    

# --- 2. Introducción del software en CLI y advertencias correspondientes ---

print(src.utils.mensaje_introduccion())   

# --- 3. Carga de historial/cache de variables globales ---


"""
json_file = ANACRONICO_DIR / "cache" / "history.json"

if json_file.exists():
    print(f"\n🔄 Cargando historial desde {json_file}...")
    history_cache_df = json.load(open(json_file, 'r'))
else: 
    print(f"❌ No se encontró el archivo de historial en {json_file}. Se crearán valores por defecto.")
    history = {
        "_comentario": ("Almacenamiento de variables globales, funciones y scripts "
                        "que el software ya ha utilizado. Por ejemplo, los que "
                        "configuran la estructura de directios, o los que crean "
                        "los entornos virtuales."),
        "version": "0.1"
    }
    """