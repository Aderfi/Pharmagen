import sys
import subprocess
from pathlib import Path
import json

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
        
        # Rutas al entorno y archivo de Conda
        self.venv_path = self.dir_proyecto.parent / "Pharmagen_Env"
        self.conda_env_file = self.dir_proyecto.parent / "Pharmagen_Env" / "conda_env_pharmagen.yml"
        
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
            self.dir_proyecto / "docs",
            self.dir_proyecto / "logs",
        ]
        for path in directorios:
            path.mkdir(parents=True, exist_ok=True)
            print(f"Directorio creado: {path}")
        print("Estructura de directorios creada correctamente.")
    
    def instalar_dependencias_python_env(self, libs=None):
        #  Verifica e instala las librerías usando el pip del entorno virtual.
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

    def instalar_dependencias_conda_env(self):
        #  Verifica e instala las librerías usando el entorno Conda.
        print("\n--- Creando o actualizando entorno Conda ---")
        log_file_path = self.dir_proyecto / "logs" / "conda_setup.log"
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.conda_env_file.exists():
            print(f"ERROR: No se encontró el archivo de entorno Conda en {self.conda_env_file}")
            return
        
        try:
            print(f"El proceso se guardará en '{log_file_path}'")
            subprocess.run(
                f"conda env create -f {self.conda_env_file} --verbose > {log_file_path} 2>&1",
                check=True, shell=True, text=True
            )
            print("Entorno Conda creado correctamente.")
        except subprocess.CalledProcessError:
            print("El entorno ya parece existir. Intentando actualizar...")
            try:
                subprocess.run(
                    ["conda", "env", "update", "--file", str(self.conda_env_file), "--prune"], 
                    check=True
                )
                print("Entorno Conda actualizado correctamente.")
            except subprocess.CalledProcessError as e:
                print(f"Error al actualizar el entorno Conda: {e}")

    def crear_config_files(self):
        #  Crea archivos de configuración iniciales si no existen.
        print("\n--- Creando archivos de configuración ---")
        config_json_file = self.dir_proyecto / "config.json"
        if not config_json_file.exists(): # config.json
            config_data = {
                "_comentario": [
                    "Almacenamiento de variables globales, funciones y scripts", 
                    "que el software ya ha utilizado."
                ],
                "data_path": str(self.dir_proyecto / "data"),
                "results_path": str(self.dir_proyecto / "results"),
                "cache_path": str(self.dir_proyecto / "cache"),
                "version": "0.1"
            }
            with open(config_json_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            print(f"Archivo de configuración creado: {config_json_file}")
        else:
            print(f"El archivo {config_json_file} ya existe.")
        
        paths_file = self.dir_proyecto / "cache" / "paths.json"
        if not paths_file.exists(): # paths.json
            paths_data = {
                "_comentario": ["Almacenamiento de rutas de uso frecuente."],
                "proyecto": str(self.dir_proyecto),
                "raw_data": str(self.dir_proyecto / "data" / "raw"),
                "processed_data": str(self.dir_proyecto / "data" / "processed"),
            }
            with open(paths_file, 'w') as f:
                json.dump(paths_data, f, indent=4)
            print(f"Archivo de rutas creado: {paths_file}")
        else:
            print(f"El archivo {paths_file} ya existe.")
    
    def ejecutar_configuracion_completa(self):
        """Ejecuta todos los pasos de la configuración inicial."""
        self.crear_directorios()
        self.crear_config_files()
        self.instalar_dependencias_conda_env()
        # self.instalar_dependencias_python_env() # Opcional si usas Conda
        print("\n✅ Configuración del entorno completada.")