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

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from src.cfg.config import LOGS_DIR, METADATA, PROJECT_ROOT, VERSION

CONFIG_FILE = Path("asaver")

def welcome_message():
    msg = f"""
    ============================================
            PHARMAGEN v{VERSION}
    ============================================
    Software para farmacogenética y deep learning.
    
    Logs: {LOGS_DIR}
    ============================================
    """
    print(msg)

def check_system_config() -> Dict[str, Any]:
    """Carga o crea el archivo de estado del sistema."""
    CONFIG_FILE = Path("asaver")
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass # Si falla, recreamos
            
    # Default Config
    config = {
        "version": VERSION,
        "os": os.name,
        "setup_completed": False
    }
    save_system_config(config)
    return config

def save_system_config(config: Dict[str, Any]):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def is_venv_active() -> bool:
    """Detecta si estamos corriendo dentro de un entorno virtual (Venv o Conda)."""
    # Check standard venv
    is_base = (sys.prefix == sys.base_prefix)
    # Check Conda
    is_conda = 'CONDA_DEFAULT_ENV' in sys.modules.get('os').environ # type: ignore
    
    return (not is_base) or is_conda

def check_environment_and_setup():
    """
    Verifica el estado del entorno antes de arrancar la aplicación.
    Si es la primera vez, lanza el setup interactivo.
    """
    # 1. Verificar si es la primera ejecución (Flag File)
    flag_file = PROJECT_ROOT / "src" / "cfg" / "venv_setup_true"
    
    if not flag_file.exists():
        print("\n" + "!"*60)
        print(" ⚠️  PRIMERA EJECUCIÓN DETECTADA O ENTORNO NO CONFIGURADO")
        print("!"*60)
        print("\nEs necesario configurar los directorios y dependencias.")
        input(">> Pulse [ENTER] para iniciar el asistente de configuración (setup.py)...")
        
        try:
            # Importación dinámica para evitar errores circulares o si setup no existe
            # Como main.py añade PROJECT_ROOT al path, esto debería funcionar
            import setup
            setup.main()
        except ImportError:
            print("❌ Error: No se encontró 'setup.py' en la raíz del proyecto.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error crítico durante el setup: {e}")
            sys.exit(1)
            
        print("\n✅ Setup finalizado. Iniciando Pharmagen...\n")
        time.sleep(1)

    # 2. Verificar Versión de Python (Warning)
    if sys.version_info < (3, 10) or sys.version_info >= (3, 11):
        print(f"⚠️  ADVERTENCIA: Estás usando Python {sys.version_info.major}.{sys.version_info.minor}.")
        print("   Este software fue diseñado para Python 3.10. Pueden ocurrir errores.\n")
        time.sleep(2)

    # 3. Verificar Entorno Virtual Activo
    if not is_venv_active():
        print("⚠️  ADVERTENCIA: No se detectó un entorno virtual activo.")
        print("   Se recomienda ejecutar este software dentro de un entorno 'venv' o 'conda'.")
        choice = input("   ¿Desea continuar de todos modos? (s/n): ").lower()
        if choice != 's':
            sys.exit(0)

def print_gnu_notice():
    """Imprime el aviso legal"""
    
    # Lógica inteligente de años
    start_year = 2025
    current_year = datetime.now().year
    
    if current_year > start_year:
        # Si estamos en 2026 o futuro, muestra "2025-2026"
        year_str = f"{start_year}-{current_year}"
    else:
        # Si estamos en 2025, muestra solo "2025"
        year_str = str(start_year)

    author = "Adrim Hamed Outmani (@Aderfi)"
    program = METADATA.get("project_name", "Pharmagen")
    
    notice = f"""
    {program} Copyright (C) {year_str} {author}
    This program comes with ABSOLUTELY NO WARRANTY; for details type `show w'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `show c' for details.
    """
    print(notice)

def print_warranty_details():
    """Texto completo para 'show w'."""
    print("\n" + "="*60)
    print("NO WARRANTY")
    print("="*60)
    print("""
    BECAUSE THE PROGRAM IS LICENSED FREE OF CHARGE, THERE IS NO WARRANTY
    FOR THE PROGRAM, TO THE EXTENT PERMITTED BY APPLICABLE LAW. EXCEPT WHEN
    OTHERWISE STATED IN WRITING THE COPYRIGHT HOLDERS AND/OR OTHER PARTIES
    PROVIDE THE PROGRAM "AS IS" WITHOUT WARRANTY OF ANY KIND, EITHER EXPRESSED
    OR IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
    MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE ENTIRE RISK AS
    TO THE QUALITY AND PERFORMANCE OF THE PROGRAM IS WITH YOU. SHOULD THE
    PROGRAM PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING,
    REPAIR OR CORRECTION.
    """)
    input("\nPresione [Enter] para volver...")

def print_conditions_details():
    """Texto completo para 'show c'."""
    print("\n" + "="*60)
    print("REDISTRIBUTION CONDITIONS")
    print("="*60)
    print("""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
    """)
    input("\nPresione [Enter] para volver...")

if __name__ == "__main__":
    welcome_message()