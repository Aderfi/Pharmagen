# Software: pharmagen_pmodel
# Versión: 0.1
# Autor: Astordna / Aderfi / Adrim Hamed Outmani
# Fecha: 2024-06-15
# Descripción: Este software tiene la utilidad de actuar como puente para los inputs y la interpretación
#              de los outputs de un modelo predictivo cuya finalidad es inferir a partir de datos genómicos
#              del paciente, la eficacia terapéutica y el posible riesgo incrementado de ciertas toxicidades.
#              El modelo predictivo está basado en un conjunto de modelos de machine learning entrenados
#              con datos genómicos y clínicos de pacientes reales.

import sys
from pathlib import Path
import json
import Pharmagen

# --- 1. Configuración de Rutas e Importación y About---

# El directorio 'Master' es la raíz desde donde se ejecuta el script.
# Python lo añade automáticamente al path, por lo que no es necesario sys.path.append.
PROJECT_ROOT = Path(__file__).resolve().parent
ANACRONICO_DIR = PROJECT_ROOT / "Anacronico"
AUTOR = "Astordna/Aderfi/Adrim Hamed Outmani"
VERSION = "0.1"
FECHA_CREACION = "2024-06-15"

with open(ANACRONICO_DIR / "cache" / "paths.json", 'r') as f:
    paths_df = json.load(f)
    


# --- 2. Introducción del software en CLI y advertencias correspondientes ---
def mostrar_introduccion():
    introduccion = f""""
    ============================================
            pharmagen_pmodel {VERSION}
    ============================================
    Autor: Astordna / Aderfi / Adrim Hamed Outmani
    Fecha: 2024-06-15
    
    \t\t\t**ADVERTENCIA IMPORTANTE**
    
    Para asegurar el correcto funcionamiento del software y evitar errores,
    es preciso ejecutar primero el archivo "Create_CONDA_ENV.py" o 
    "CREATE_VENV.py" ubicado en la carpeta Environment_Scripts.
    
    Esto creará el entorno virtual de trabajo con las herramientas y librerías necesarias.
    ============================================
    """
    return introduccion

# --- 3. Carga de historial/cache de variables globales ---

print(mostrar_introduccion())

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