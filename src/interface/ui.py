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

import itertools

# Agrupación de herramientas visuales genéricas
import sys
import threading
import time
from pathlib import Path
from typing import Optional


import sys
import threading
import time
import itertools
from pathlib import Path
from typing import Optional

class Spinner:
    """
    Context Manager for console loading animations.
    """
    def __init__(self, message: str = "Processing..."):
        self.message = message
        self.stop_running = False
        self.spinner_chars = itertools.cycle(['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'])
        self.thread: Optional[threading.Thread] = None

    def _spin(self):
        while not self.stop_running:
            sys.stdout.write(f"\r{next(self.spinner_chars)} {self.message}")
            sys.stdout.flush()
            time.sleep(0.1)
        # Clear line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')
        sys.stdout.flush()

    def __enter__(self):
        self.thread = threading.Thread(target=self._spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_running = True
        if self.thread:
            self.thread.join()

class ConsoleIO:
    """
    Static helper for Console Input/Output operations.
    """
    @staticmethod
    def print_header(title: str):
        print("\n" + "═" * 60)
        print(f" {title}")
        print("═" * 60)

    @staticmethod
    def print_success(msg: str):
        print(f"✅ {msg}")

    @staticmethod
    def print_error(msg: str):
        print(f"❌ {msg}")

    @staticmethod
    def input_path(prompt: str, default: Optional[Path] = None, must_exist: bool = True) -> Path:
        while True:
            default_str = f" [{default}]" if default else ""
            path_str = input(f"{prompt}{default_str}: ").strip()
            
            if not path_str and default:
                return default
            
            if not path_str:
                ConsoleIO.print_error("A path is required.")
                continue
                
            path = Path(path_str)
            if must_exist and not path.exists():
                ConsoleIO.print_error(f"Path does not exist: {path}")
            else:
                return path