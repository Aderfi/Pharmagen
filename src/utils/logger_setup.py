# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning
# Copyright (C) 2025  Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

from datetime import datetime
import logging

from src.cfg.manager import DIRS


def setup_logging(name="Pharmagen", level=logging.INFO) -> None:
    log_file = DIRS["logs"] / f"{name}_{datetime.now():%Y-%m-%d}.log"

    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("").addHandler(console)

    # Silence noise
    for lib in ["matplotlib", "optuna", "numba"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
