# Copyright (C) 2023 [Tu Nombre / Pharmagen Team]
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

# Agrupación de herramientas visuales genéricas
import sys
import time
import threading
import itertools
from pathlib import Path
from typing import Optional

class Spinner:
    """
    Context Manager para animaciones de carga en consola.
    Permite envolver procesos largos con una animación visual.
    """
    def __init__(self, message="Procesando..."):
        self.message = message
        self.stop_running = False
        # Ciclo de caracteres para el spinner
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])

    def spin(self):
        while not self.stop_running:
            sys.stdout.write(f"\r{next(self.spinner_chars)} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Limpiar línea al terminar
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')

    def __enter__(self):
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        self.thread.join()

def input_path(prompt: str, default: Optional[Path] = None, must_exist: bool = True) -> Path:
    """
    Solicita una ruta al usuario con validación.
    """
    while True:
        default_str = f" [{default}]" if default else ""
        path_str = input(f"{prompt}{default_str}: ").strip()
        
        # Usar default si el usuario da Enter vacío
        if not path_str and default:
            return default
        
        if not path_str and not default:
            print("❌ Debe ingresar una ruta.")
            continue
            
        path = Path(path_str)
        
        if must_exist and not path.exists():
            print(f"❌ El archivo/directorio no existe: {path}")
        else:
            return path

def print_header(title: str):
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_success(msg: str):
    print(f"✅ {msg}")

def print_error(msg: str):
    print(f"❌ {msg}")