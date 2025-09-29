import sys
from pathlib import Path

# --- 1. Configuración de Rutas e Importación ---

# El directorio 'Master' es la raíz desde donde se ejecuta el script.
# Python lo añade automáticamente al path, por lo que no es necesario sys.path.append.
PROJECT_ROOT = Path(__file__).resolve().parent
ANACRONICO_DIR = PROJECT_ROOT / "Anacronico"

try:
    # La importación correcta refleja la estructura de carpetas: Anacronico -> scripts
    from Anacronico.scripts import ConfiguradorEntorno
except ImportError as e:
    print(f"ERROR: No se pudo importar 'ConfiguradorEntorno'.")
    print(f"Asegúrate de que la estructura 'Master/Anacronico/scripts' es correcta.")
    print(f"Detalle del error: {e}")
    sys.exit(1)

# --- 2. Funciones Principales de la Aplicación ---

def programa_principal():
    """
    Contiene la lógica principal de la aplicación.
    """
    print("\n--- Iniciando lógica principal del programa ---")
    # from Anacronico.scripts import procesar_genomas
    # procesar_genomas(ANACRONICO_DIR)
    print("Análisis completado.")


# --- 3. Punto de Entrada del Programa ---
if __name__ == "__main__":
    
    print("--- Iniciando configuración del entorno ---")
    # Pasamos la ruta a la carpeta 'Anacronico' al configurador
    configurador = ConfiguradorEntorno(ANACRONICO_DIR)
    configurador.ejecutar_configuracion()
    
    programa_principal()