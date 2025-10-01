import json
import sys
import os
import subprocess
from Pharmagen.config import *
import shutil

def mensaje_introduccion():
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
            "environment_created": bool(0),
            "dirs_created": bool(0),
            "libs_installed": bool(0),
            "date_database": "",
            "version": VERSION,
            "os": f"{os.name}"  # "NT (Windows)" o "Posix (Linux)"
        }

    return config_df

def check_config(config_df, choice):
    if config_df["environment_created"] is False:
        print(f"\n⚠️  El entorno virtual no ha sido creado. Se va a ejecutar el script \
            para crearlo. \
            \n Por favor, escribe 1 para hacerlo a través de Conda o 2 para hacerlo con venv.")
        
        try:
            if [choice == '1'] and (sys.platform == 'win32'):
                print("\nEjecutando Create_CONDA_ENV.py...")
                os.system(f'python "{ENV_SCRIPTS_DIR}/Create_CONDA_ENV.py"')

            elif [choice == '1'] and (sys.platform != 'linux'): # Ejecutar Create_CONDA_ENV.py en Linux/Mac
                print("\nEjecutando Create_CONDA_ENV.py...")
                (f'python "{ENV_SCRIPTS_DIR}/Create_CONDA_ENV.py"')
            
            elif [choice == '2']:
                print("\nEjecutando CREATE_VENV.py...")
                os.system(f'python "{ENV_SCRIPTS_DIR}/CREATE_VENV.py"')
        
        except Exception as e:
            print(f"Error al intentar crear el entorno virtual: {e}")
            return
    return