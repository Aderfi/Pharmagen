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
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union, Set, Dict, Any, Tuple

# Intento de importación segura para rapidfuzz
try:
    from rapidfuzz import process, fuzz
except ImportError:
    process = None
    fuzz = None

from src.cfg.config import DICTS_DATA_DIR

logger = logging.getLogger(__name__)

# Pre-compilación de Regex
RE_SPLITTERS = re.compile(r"[,;|/]+")
RE_SPACES = re.compile(r"\s+")
UNKNOWN_TOKEN = "__UNKNOWN__"

def serialize_multi_labelcols(label_input: Any) -> str:
    """Serializa entradas a string separado por pipes."""
    if label_input is None or (isinstance(label_input, float) and np.isnan(label_input)) or label_input == "":
        return ""

    parts: Set[str] = set()
    if isinstance(label_input, str):
        parts = {s.strip() for s in RE_SPLITTERS.split(label_input) if s.strip()}
    elif isinstance(label_input, (list, tuple, np.ndarray)):
        parts = {str(x).strip() for x in label_input if x is not None and str(x).strip()}
    
    return "|".join(sorted(parts))

def load_atc_dictionary(json_path: Union[str, Path], lang_idx: int = 1) -> Dict[str, str]:
    """Carga diccionario ATC."""
    path = Path(json_path)
    if not path.exists():
        logger.warning(f"Diccionario ATC no encontrado en {path}.")
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            atc_raw = json.load(f)
        return {v[lang_idx].lower(): k for k, v in atc_raw.items()}
    except Exception as e:
        logger.error(f"Error crítico leyendo JSON ATC: {e}")
        return {}

def drugs_to_atc(
    df: pd.DataFrame, 
    drug_col: str, 
    atc_dict_rev: Dict[str, str],
    atc_col: str = "atc"
) -> pd.DataFrame:
    """Mapea fármacos a ATC con fuzzy matching."""
    if not atc_dict_rev:
        df[atc_col] = "No_ATC"
        return df

    atc_keys = list(atc_dict_rev.keys())
    memoization: Dict[str, str] = {}

    def get_atc_codes(text: Any) -> str:
        if not isinstance(text, str) or not text: return "No_ATC"
        text_lower = text.lower().strip()
        
        if text_lower in memoization: return memoization[text_lower]

        tokens = [t.strip() for t in RE_SPLITTERS.split(text_lower) if t.strip()]
        codes = []

        for token in tokens:
            if token in atc_dict_rev:
                codes.append(atc_dict_rev[token])
            elif process: # Fuzzy match solo si rapidfuzz está instalado
                result = process.extractOne(token, atc_keys, scorer=fuzz.WRatio, score_cutoff=92)
                if result:
                    codes.append(atc_dict_rev[result[0]])
                else:
                    codes.append("No_ATC")
            else:
                codes.append("No_ATC")

        valid_codes = [c for c in codes if c != "No_ATC"]
        final_code = "|".join(sorted(set(valid_codes))) if valid_codes else "No_ATC"
        memoization[text_lower] = final_code
        return final_code

    logger.info(f"Mapeando columna '{drug_col}' a ATC...")
    df[atc_col] = df[drug_col].map(get_atc_codes).astype("string")
    return df

def normalize_dataframe(
    df: pd.DataFrame,
    all_cols: List[str],
    multi_label_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Limpieza y normalización estándar del DataFrame."""
    multi_label_cols = multi_label_cols or []
    
    # Normalizar nombres de columnas
    df.columns = df.columns.str.lower().str.strip()
    
    all_cols_lower = [c.lower() for c in all_cols]
    multi_label_lower = [c.lower() for c in multi_label_cols]
    
    cols_present = set(df.columns)
    single_label_cols = [c for c in cols_present if c in all_cols_lower and c not in multi_label_lower]

    # Limpieza Single Label
    if single_label_cols:
        df[single_label_cols] = df[single_label_cols].fillna("UNKNOWN").astype(str)
        for col in single_label_cols:
            df[col] = (
                df[col].str.replace(RE_SPLITTERS, "|", regex=True)
                       .str.replace(" ", "_", regex=False)
                       .str.strip().str.lower()
            )

    # Limpieza Multi Label
    for col in multi_label_lower:
        if col in cols_present:
            df[col] = df[col].apply(serialize_multi_labelcols).str.lower()
            
    return df