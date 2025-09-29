import json
from Pharmagen.config import *

def mensaje_introduccion(self=None):
    introduccion = f""""
    ============================================
            pharmagen_pmodel {VERSION}
    ============================================
    Autor: {AUTOR}
    Fecha: {FECHA_CREACION}
    Descripción: Software para predecir la eficacia terapéutica y toxicidades en base a datos genómicos
                 y el entrenamiento de un modelo predictivo de machine learning.
                 
    ============================================

    \t\t\t**ADVERTENCIA IMPORTANTE**
    
    Para asegurar el correcto funcionamiento del software y evitar errores,
    es preciso ejecutar primero el archivo "Create_CONDA_ENV.py" o 
    "CREATE_VENV.py" ubicado en la carpeta Environment_Scripts.
    SOLO SI ES LA PRIMERA VEZ QUE EJECUTAS EL SOFTWARE.
    
    
    Esto creará el entorno virtual de trabajo con las herramientas y librerías necesarias.
    ============================================
    
    Todos los errores y logs se almacenarán en:
    \t\t\t\t{LOGS_DIR}
    
    """
    return introduccion

def load_config():
    json_file = CONFIG_FILE

    if json_file.exists():
        print(f"\n\t\t>>>Cargando historial desde {json_file}...")
        config_df = json.load(open(json_file, 'r'))
    else: 
        print(f"\t\tNo se encontró el archivo de historial en {json_file}. Se creará con las estructuras \
              por defecto.")
        config_df = {
            "_comentario": [
                "Almacenamiento de variables globales, funciones y scripts", 
            "que el software ya ha utilizado. Por ejemplo, los que configuran", 
            "la estructura de directorios, o los que crean los entornos virtuales."
            ],
            "environment_created": 0,
            "dirs_created": 0,
            "libs_installed": 0,
            "date_database": "",
            "version": "0.1"
        }

     