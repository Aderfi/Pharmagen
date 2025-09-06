"""
import os
import sys
import subprocess

# Creación de la estructura de directorios
def creacion_workspace(MAIN_DIR):
    subdirs = ['data', 'results', 'cache', 'docs']
    
    for subdir in subdirs:
        path = os.path.join(MAIN_DIR, subdir)
        os.makedirs(path, exist_ok=True)
        print(f"Directorio creado o ya existente: {path}")

    if not os.listdir(os.path.join(MAIN_DIR, 'data', 'raw')):
        os.makedirs(os.path.join(MAIN_DIR, 'data', 'raw'), exist_ok=True)
    elif not os.listdir(os.path.join(MAIN_DIR, 'data', 'processed')):
        os.makedirs(os.path.join(MAIN_DIR, 'data', 'processed'), exist_ok=True)
print("\n\nEstructura de directorios creada")

            
# Activación del entorno virtual
script_dir = os.path.dirname(__file__)

if os.name == 'nt':
    # Ruta al python.exe dentro del venv en Windows
    python_executable = os.path.join(script_dir, '..', '..', 'Environment', 'DeepEnvMaster', 'Scripts', 'python.exe')
elif os.name == 'posix':
    # Ruta al python dentro del venv en Linux/macOS
    python_executable = os.path.join(script_dir, '..', '..', 'Environment', 'DeepEnvMaster', 'bin', 'python')

if not os.path.exists(python_executable):
    print(f"ERROR: No se encontró el ejecutable de Python en {python_executable}")
else:
    pass    
    
# Verificación e instalación de librerías necesarias
def librerias_necesarias(python_exe):
    libs = ["pandas", "biopython"]
    for lib in libs:    
        try:
            # Usamos subprocess para tener más control y capturar errores
            subprocess.run([python_exe, "-m", "pip", "show", lib], check=True, capture_output=True)
            print(f"\tLa librería '{lib}' ya está instalada.")
        except subprocess.CalledProcessError:
            print(f"\nLa librería '{lib}' no está instalada. Instalándola ahora...")
            subprocess.run([python_exe, "-m", "pip", "install", lib], check=True)
"""
import sys
import subprocess
from pathlib import Path

class ConfiguradorEntorno:
    """
    Configura el entorno del proyecto: crea la estructura de directorios
    y verifica/instala las dependencias desde el venv correcto.
    """
    def __init__(self, dir_proyecto):
        """
        Inicializa el configurador.
        'dir_proyecto' es la ruta a la carpeta raíz del proyecto ('Anacronico').
        """
        self.dir_proyecto = Path(dir_proyecto).resolve()
        
        # Para encontrar 'Environment', subimos un nivel desde el dir del proyecto ('Anacronico' -> 'Master')
        # y luego bajamos a 'Environment/DeepEnvMaster'.
        self.venv_path = self.dir_proyecto.parent / "Environment" / "DeepEnvMaster"
        
        # Determina la ruta al ejecutable de Python del entorno virtual
        if sys.platform == "win32":
            self.python_executable = self.venv_path / "Scripts" / "python.exe"
        else:
            self.python_executable = self.venv_path / "bin" / "python"
        
        print(f"Directorio del proyecto: {self.dir_proyecto}")
        print(f"Ruta del ejecutable de Python del venv: {self.python_executable}")

    def crear_directorios(self):
        """Crea la estructura de directorios necesaria dentro del proyecto."""
        print("\n--- Creando estructura de directorios ---")
        directorios = [
            self.dir_proyecto / "data" / "raw",
            self.dir_proyecto / "data" / "processed",
            self.dir_proyecto / "results",
            self.dir_proyecto / "cache",
            self.dir_proyecto / "docs"
        ]
        for path in directorios:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directorio asegurado: {path}")

    def instalar_dependencias(self, libs=None):
        """Verifica e instala las librerías usando el pip del entorno virtual."""
        if libs is None:
            libs = ["pandas", "biopython"]
            
        print("\n--- Verificando dependencias ---")
        if not self.python_executable.exists():
            print(f"ERROR: No se encontró el ejecutable de Python en {self.python_executable}")
            print("Asegúrate de que la estructura de carpetas es correcta.")
            return

        for lib in libs:
            try:
                subprocess.run(
                    [str(self.python_executable), "-m", "pip", "show", lib], 
                    check=True, capture_output=True, text=True
                )
                print(f"'{lib}' ya está instalada.")
            except subprocess.CalledProcessError:
                print(f"Instalando '{lib}'...")
                subprocess.run(
                    [str(self.python_executable), "-m", "pip", "install", lib], 
                    check=True
                )

    def ejecutar_configuracion(self):
        """Ejecuta todos los pasos de la configuración."""
        self.crear_directorios()
        self.instalar_dependencias()
        print("\n✅ Configuración del entorno completada.")

# --- Bloque para ejecución directa (testing) ---
if __name__ == "__main__":
    # Cuando se ejecuta este script directamente, el dir. del proyecto está un nivel arriba.
    directorio_proyecto_anacronico = Path(__file__).resolve().parent.parent
    
    configurador = ConfiguradorEntorno(directorio_proyecto_anacronico)
    configurador.ejecutar_configuracion()