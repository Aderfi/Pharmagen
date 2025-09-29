import sys
from pathlib import Path

# Añadimos la ruta raíz del proyecto (un nivel arriba de 'scripts') para poder importar desde 'src'
# Esto hace que el script funcione sin importar desde dónde lo llames
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.project_config import ConfiguradorEntorno

def main():
    print("--- CONFIGURACIÓN DEL PROYECTO Pharmagen ---")
    
    # La raíz del proyecto ('Pharmagen') es el directorio padre de la carpeta 'scripts'
    directorio_proyecto_anacronico = project_root
    
    configurador = ConfiguradorEntorno(directorio_proyecto_anacronico)
    configurador.ejecutar_configuracion_completa()

if __name__ == "__main__":
    main()