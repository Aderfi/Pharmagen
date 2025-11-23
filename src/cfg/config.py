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

from datetime import datetime
from pathlib import Path

import tomli  # Requiere: pip install tomli Python <= 3.10 | pip install tomllib Python >= 3.11

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PATHS_FILE = Path(__file__).parent / "paths.toml"
CFG_FILE = Path(__file__).parent / "config.toml"

if not PATHS_FILE.exists() or not CFG_FILE.exists():
    raise FileNotFoundError(f"CRITICAL: paths.toml or config.toml not found at {PATHS_FILE} and {CFG_FILE}")

with open(PATHS_FILE, "rb") as f_paths, open(CFG_FILE, "rb") as f_cfg:
    _PATHS_CONFIG = tomli.load(f_paths  )
    _GLOBAL_CONFIG = tomli.load(f_cfg)

def _resolve_path(relative_path: str) -> Path:
    return PROJECT_ROOT / relative_path

# Metadatos
METADATA = _GLOBAL_CONFIG.get("metadata", {})
PROJECT_NAME = METADATA.get("project_name", "Pharmagen")
VERSION = METADATA.get("version", "Couldnt fetch version")

# Directorios Base
_base = _PATHS_CONFIG.get("base", {})
DATA_DIR = _resolve_path(_base.get("data", "data"))
LOGS_DIR = _resolve_path(_base.get("logs", "logs"))
RESULTS_DIR = _resolve_path(_base.get("results", "results"))
REPORTS_DIR = _resolve_path(_base.get("reports", "reports"))
CACHE_DIR = _resolve_path(_base.get("cache", "cache"))
SRC_DIR = _resolve_path(_base.get("src", "src"))

# Subdirectorios de Data
_data = _PATHS_CONFIG.get("data", {})
RAW_DATA_DIR = _resolve_path(_data.get("raw", "data/raw"))
PROCESSED_DATA_DIR = _resolve_path(_data.get("processed", "data/processed"))
DICTS_DATA_DIR = _resolve_path(_data.get("dicts", "data/dicts"))
REF_GENOME_DIR = _resolve_path(_data.get("ref_genome", "data/ref_genome"))

# Datos de entrenamiento
MODEL_TRAIN_DATA = _resolve_path(_data.get("train", "train_data"))

# Subdirectorios de Modelos
_models = _PATHS_CONFIG.get("models", {})
PGEN_MODEL_DIR = _resolve_path(_models.get("root", "src/pgen_model"))
MODELS_DIR = _resolve_path(_models.get("models_saved", "src/pgen_model/models"))
MODEL_ENCODERS_DIR = _resolve_path(_models.get("encoders", "src/pgen_model/encoders"))
MODEL_VOCABS_DIR = _resolve_path(_models.get("vocabs", "src/pgen_model/labels_vocabs"))

# Subdirectorios de Reports
_reports = _PATHS_CONFIG.get("reports", {})
OPTUNA_REPORTS_DIR = _resolve_path(_reports.get("optuna", "reports/optuna_reports"))
FIGURES_DIR = _resolve_path(_reports.get("figures", "reports/figures"))

# Archivos Específicos (Helpers)
REF_GENOME_FASTA = REF_GENOME_DIR / "HSapiens_GChr38.fa"
LOG_FILE = LOGS_DIR / "pharmagen_runtime.log"

MULTI_LABEL_COLUMN_NAMES = _PATHS_CONFIG.get("multi_label_column_names", ["Phenotype_outcome"])
DATE_STAMP = datetime.now().strftime("%Y_%m_%d")


# Crear directorios si no existen
for path in [DATA_DIR, LOGS_DIR, RESULTS_DIR, MODELS_DIR, CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Version: {VERSION}")

    print(f"Data Dir: {DATA_DIR}")
    print(f"Logs Dir: {LOGS_DIR}")
    print(f"Models Dir: {MODELS_DIR}")
    print(f"Model Train Data: {MODEL_TRAIN_DATA}")