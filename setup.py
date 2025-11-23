# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#!/usr/bin/env python3
# coding=utf-8
"""
Script de Inicialización y Configuración de Entorno (Bootstrap).

Este módulo es responsable de preparar el sistema operativo para ejecutar Pharmagen.
Se debe ejecutar una única vez tras clonar el repositorio o si se necesita
reparar la estructura del proyecto.

Funcionalidades:
    - Verificación de la versión de Python (Target: 3.10).
    - Creación de la estructura de directorios estándar (data, logs, models, etc.).
    - Detección e instalación automática de dependencias (pip/conda).
    - Generación de archivos de bandera (flag files) para indicar estado saludable.

Uso:
    $ python setup.py

Author:
    Adrim Hamed Outmani (@Aderfi)
"""
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

# Configuración de Logging para el Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - SETUP - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("PharmagenSetup")

# Constantes
PROJECT_ROOT = Path(__file__).resolve().parent
SETUP_FLAG_FILE = PROJECT_ROOT / "src" / "cfg" / "venv_setup_true"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
PYTHON_TARGET = (3, 10)

class ProjectSetup:
    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_dir = self.root / "data"
        self.src_dir = self.root / "src"
        self.logs_dir = self.root / "logs"

    def _check_python_version(self):
        """Verifica que se esté ejecutando con la versión correcta."""
        ver = sys.version_info
        if ver.major != PYTHON_TARGET[0] or ver.minor != PYTHON_TARGET[1]:
            logger.warning(
                f"⚠️ Estás ejecutando el setup con Python {ver.major}.{ver.minor}. "
                f"Se recomienda Python {PYTHON_TARGET[0]}.{PYTHON_TARGET[1]}."
            )
        else:
            logger.info(f"✅ Versión de Python correcta: {ver.major}.{ver.minor}")

    def create_directories(self):
        """Crea la estructura de carpetas necesaria."""
        logger.info("Creando estructura de directorios...")
        
        dirs_to_create = [
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "dicts",
            self.data_dir / "ref_genome",
            self.root / "results",
            self.root / "cache",
            self.root / "reports" / "figures",
            self.root / "reports" / "optuna_reports",
            self.logs_dir,
            self.src_dir / "cfg",
            self.src_dir / "pgen_model" / "models",
            self.src_dir / "pgen_model" / "encoders",
        ]

        for d in dirs_to_create:
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"No se pudo crear {d}: {e}")

        logger.info("✅ Estructura de directorios verificada.")

    def setup_dependencies(self, mode: Literal["venv", "conda", "skip"]):
        """Instala las dependencias según el modo elegido."""
        if mode == "skip":
            logger.info("Saltando instalación de dependencias.")
            return

        logger.info(f"Iniciando instalación de dependencias (Modo: {mode})...")
        
        if not REQUIREMENTS_FILE.exists():
            logger.error(f"❌ No se encuentra {REQUIREMENTS_FILE}")
            return

        try:
            # Detectar pip
            pip_cmd = [sys.executable, "-m", "pip"]
            
            # 1. Actualizar pip
            subprocess.run(pip_cmd + ["install", "--upgrade", "pip"], check=True)
            
            # 2. Instalar requirements
            logger.info("\t Descargando e instalando librerías (esto puede tardar)...")
            subprocess.run(pip_cmd + ["install", "-r", str(REQUIREMENTS_FILE)], check=True)
            
            logger.info("\t Dependencias instaladas correctamente.")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"\n\t Error instalando dependencias: {e}")
            sys.exit(1)
    
    def download_reference_data(self):
        """Descarga e indexa el genoma de referencia."""
        try:
            from src.analysis.ref_genome import ReferenceGenomeManager
            print("\n🧬 Preparando Genoma de Referencia...")
            manager = ReferenceGenomeManager()
            manager.run()
        except ImportError:
            print("⚠️ No se pudo importar el gestor de referencia (¿faltan dependencias?).")

    def create_flag_file(self):
        """Crea el archivo bandera para indicar que el setup finalizó."""
        try:
            # Asegurar que el directorio existe
            SETUP_FLAG_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            with open(SETUP_FLAG_FILE, "w") as f:
                f.write(f"Setup completado en {sys.version}\n")
                f.write("Delete this file to force re-setup.")
            
            logger.info(f"✅ Setup completado. Flag creada en: {SETUP_FLAG_FILE}")
        except Exception as e:
            logger.error(f"Error creando flag file: {e}")

    def run(self):
        print("\n" + "="*50)
        print(" 🛠️  PHARMAGEN INITIAL SETUP ")
        print("="*50)
        
        self._check_python_version()
        self.create_directories()

        # Interacción con usuario para dependencias
        print("\nSeleccione el gestor de entorno:")
        print("1. Pip (Entorno Virtual actual)")
        print("2. Conda (Entorno actual)")
        print("3. Omitir instalación (Solo crear directorios)")
        print("\n ===== ATENCION: Se recomienda CONDA si ejecutas en Windows.====\
                  ===== Pip es más óptimo en Linux/OSX.                      ====")
        
        choice = input("Opción [1]: ").strip()
        
        mode = "venv"
        if choice == "2": mode = "conda"
        elif choice == "3": mode = "skip"

        self.setup_dependencies(mode)
        self.create_flag_file()
        print("\n🎉 Configuración finalizada con éxito.\n")

def main():
    setup = ProjectSetup()
    setup.run()

if __name__ == "__main__":
    main()