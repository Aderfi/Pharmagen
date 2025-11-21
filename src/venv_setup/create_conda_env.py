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

import subprocess
import sys
import shutil
from pathlib import Path

# Configuración
ENV_NAME = "pharmagen_env"
PYTHON_VERSION = "3.10"
REQUIREMENTS_FILE = "req_venv.txt"

def check_conda_installed():
    """Verifica si Conda está disponible en el PATH system."""
    if not shutil.which("conda"):
        print("❌ Error: 'conda' no se encuentra instalado o no está en el PATH.")
        print("   Por favor instala Anaconda o Miniconda antes de ejecutar este script.")
        return False
    return True

def create_conda_env():
    """Crea el entorno usando comandos de conda."""
    print(f"🔧 Creando entorno Conda '{ENV_NAME}' con Python {PYTHON_VERSION}...")
    
    # 1. Crear entorno (conda create -n nombre python=3.10 -y)
    try:
        command = [
            "conda", "create", 
            "-n", ENV_NAME, 
            f"python={PYTHON_VERSION}", 
            "-y"  # Auto-confirmar
        ]
        subprocess.run(command, check=True)
        print(f"✅ Entorno '{ENV_NAME}' creado correctamente.")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Falló la creación del entorno Conda.")
        return False

def install_dependencies():
    """Instala dependencias usando pip DENTRO del entorno Conda."""
    print(f"📦 Instalando dependencias desde {REQUIREMENTS_FILE}...")
    
    req_path = Path(REQUIREMENTS_FILE)
    if not req_path.exists():
        print(f"❌ No se encontró el archivo: {req_path.absolute()}")
        return False

    # Truco: Usamos 'conda run' para ejecutar pip dentro del entorno sin activarlo manualmente
    try:
        # Primero actualizamos pip dentro del entorno
        subprocess.run(
            ["conda", "run", "-n", ENV_NAME, "pip", "install", "--upgrade", "pip"],
            check=True
        )
        # Instalamos requirements
        subprocess.run(
            ["conda", "run", "-n", ENV_NAME, "pip", "install", "-r", str(req_path)],
            check=True
        )
        print("✅ Dependencias instaladas exitosamente.")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error instalando dependencias.")
        return False

def main():
    print("=== Configuración de Entorno CONDA para Pharmagen ===")
    
    if not check_conda_installed():
        sys.exit(1)
        
    if not create_conda_env():
        sys.exit(1)
        
    if not install_dependencies():
        print("⚠️ El entorno se creó, pero fallaron las dependencias.")
        sys.exit(1)

    print("\n🎉 ¡Instalación Completa!")
    print("Para activar el entorno ejecuta:")
    print(f"    conda activate {ENV_NAME}")

if __name__ == "__main__":
    main()