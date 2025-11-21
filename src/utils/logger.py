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

import logging
from datetime import datetime
from pathlib import Path
from src.cfg.config import LOGS_DIR

def setup_logging():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"Pharmagen_{datetime.now().strftime('%Y-%m-%d')}.log"
    
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para errores críticos en consola
    console = logging.StreamHandler()
    console.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # Reducir ruido de librerías de terceros
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("optuna").setLevel(logging.WARNING)