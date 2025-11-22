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

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Configuración
TARGET_VERSION = "3.10" 
ENV_DIR = "venv_pharmagen"
REQUIREMENTS_FILE = "req_venv.txt"

def find_python_executable(target_version):
    """
    Intenta encontrar un ejecutable de Python que coincida con la versión target.
    Busca 'python3.10', 'python', 'py -3.10', etc.
    """
    print(f"🔍 Buscando Python {target_version} en el sistema...")

    # Lista de candidatos a probar
    candidates = [
        f"python{target_version}",  # Linux/Mac standard
        "python",                   # Default
        "python3",
        "python3.10",
        sys.executable              # El que está corriendo este script
    ]
    
    # En Windows, probamos el launcher 'py'
    if sys.platform == "win32":
        # Check especial para el launcher de windows
        try:
            res = subprocess.run(["py", f"-{target_version}", "--version"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0:
                return ["py", f"-{target_version}"] # Retornamos como lista de comando
        except FileNotFoundError:
            pass

    for cmd in candidates:
        if isinstance(cmd, str) and not shutil.which(cmd):
            continue
            
        try:
            # Verificar versión
            cmd_list = [cmd] if isinstance(cmd, str) else cmd
            res = subprocess.run(cmd_list + ["--version"], 
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            output = res.stdout + res.stderr
            
            if f"Python {target_version}" in output:
                print(f"✅ Encontrado: {' '.join(cmd_list)} ({output.strip()})")
                return cmd_list
        except Exception:
            continue

    print(f"❌ No se encontró una instalación de Python {target_version}.")
    print("Por favor instala Python 3.10 desde python.org")
    return None

def create_venv(python_cmd):
    """Crea el entorno virtual."""
    if Path(ENV_DIR).exists():
        print(f"⚠️ El directorio '{ENV_DIR}' ya existe. Borrando...")
        shutil.rmtree(ENV_DIR)
        
    print(f"🔧 Creando entorno virtual en ./{ENV_DIR}...")
    try:
        # python -m venv venv_dir
        cmd = python_cmd + ["-m", "venv", ENV_DIR]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        print("❌ Falló la creación del entorno virtual.")
        return False

def install_dependencies():
    """Instala requirements.txt usando el pip del entorno virtual."""
    print("📦 Instalando dependencias...")
    
    # Determinar ruta de pip dentro del venv
    if sys.platform == "win32":
        pip_exe = Path(ENV_DIR) / "Scripts" / "pip.exe"
    else:
        pip_exe = Path(ENV_DIR) / "bin" / "pip"

    if not pip_exe.exists():
        print(f"❌ No se encontró pip en {pip_exe}")
        return False

    req_path = Path(REQUIREMENTS_FILE)
    if not req_path.exists():
        print(f"❌ Falta el archivo {REQUIREMENTS_FILE}")
        return False

    try:
        # Upgrade pip
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
        # Install requirements
        subprocess.run([str(pip_exe), "install", "-r", str(req_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando librerías: {e}")
        return False

def main():
    print("=== Configuración de VENV (Nativo) para Pharmagen ===")
    
    python_cmd = find_python_executable(TARGET_VERSION)
    if not python_cmd:
        sys.exit(1)
        
    if not create_venv(python_cmd):
        sys.exit(1)
        
    if not install_dependencies():
        sys.exit(1)

    print("\n🎉 ¡Instalación Completa!")
    print("Para activar el entorno:")
    if sys.platform == "win32":
        print(f"    .\\{ENV_DIR}\\Scripts\\activate")
    else:
        print(f"    source ./{ENV_DIR}/bin/activate")

if __name__ == "__main__":
    main()