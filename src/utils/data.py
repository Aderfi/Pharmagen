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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# Intento de importación segura para rapidfuzz
try:
    from rapidfuzz import fuzz, process
except ImportError:
    process = None
    fuzz = None

from src.cfg.config import DICTS_DATA_DIR

logger = logging.getLogger(__name__)

# Pre-compilación de Regex
RE_SPLITTERS = re.compile(r"[,;|]+")
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
            elif process: 
                result = process.extractOne(token, atc_keys, score_cutoff=92)
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

def load_and_prep_dataset(
    csv_path: Union[str, Path],
    all_cols: List[str],
    target_cols: List[str],
    multi_label_targets: Optional[List[str]] = None,
    stratify_cols: Union[List[str], str, None] = None,
) -> pd.DataFrame:
    """
    Carga, limpia, normaliza y filtra el dataset en un solo paso robusto.
    """
    csv_path = Path(csv_path)
    multi_label_targets = multi_label_targets or []
    
    # Normalizamos inputs de configuración a minúsculas para evitar errores
    all_cols_lower = [c.lower() for c in all_cols]
    multi_label_lower = [c.lower() for c in multi_label_targets]
    
    logger.info(f"Cargando y procesando datos desde {csv_path}...")

    # -------------------------------------------------------
    # 1. CARGA (LOAD)
    # -------------------------------------------------------
    try:
        df = pd.read_csv(
            csv_path, 
            sep="\t", 
            usecols=all_cols, 
            dtype=str
        )
    except ValueError:
        logger.warning("Error filtrando columnas en carga. Leyendo archivo completo...")
        df = pd.read_csv(csv_path, sep="\t", dtype=str)

    # -------------------------------------------------------
    # 2. ESTANDARIZACIÓN INICIAL (Initial Setup)
    # -------------------------------------------------------
    # Forzamos minúsculas y sin espacios en los nombres de las columnas YA MISMO
    df.columns = df.columns.str.lower().str.strip()

    # -------------------------------------------------------
    # 3. LIMPIEZA DE CONTENIDO (Cleaning)
    # -------------------------------------------------------
    # A. Features y Single Labels
    # Identificamos qué columnas tenemos realmente tras la carga
    existing_cols = set(df.columns)
    
    # Calculamos single labels: están en 'all_cols' pero NO son multilabel
    single_label_cols = [
        c for c in all_cols_lower 
        if c in existing_cols and c not in multi_label_lower
    ]

    if single_label_cols:
        # Relleno masivo de nulos y conversión a string
        df[single_label_cols] = df[single_label_cols].fillna("UNKNOWN").astype(str)
        
        # Limpieza vectorizada
        for col in single_label_cols:
            df[col] = (
                df[col]
                .str.replace(", ", ",")
                .str.replace(r"[,;]+", "|", regex=True) # Unificar separadores
                .str.replace(" ", "_", regex=False)     # Espacios a guiones bajos
                .str.strip()
                .str.lower()
            )

    # B. Features Multi Label
    for col in multi_label_lower:
        if col in existing_cols:
            # Asumimos que serialize_multi_labelcols maneja nulos
            df[col] = df[col].apply(serialize_multi_labelcols).str.lower()

    # -------------------------------------------------------
    # 4. LÓGICA DE DOMINIO (Domain Logic)
    # -------------------------------------------------------
    # Ejemplo: Generación de códigos ATC si tenemos fármacos
    if "drug" in df.columns and "atc" not in df.columns:
        # df = drugs_to_atc(df, drug_col="drug", atc_col="atc")
        # logger.info("Columna ATC generada.")
        pass

    # -------------------------------------------------------
    # 5. ESTRATIFICACIÓN Y FILTRADO (Stratify & Filter)
    # -------------------------------------------------------
    if stratify_cols:
        s_cols = [stratify_cols] if isinstance(stratify_cols, str) else stratify_cols
        # Normalizamos nombres de config a minúsculas
        s_cols = [s.lower() for s in s_cols]
        
        existing_strat_cols = [c for c in s_cols if c in df.columns]

        if existing_strat_cols:
            # Crear la columna combinada
            df["stratify_col"] = df[existing_strat_cols].astype(str).agg("|".join, axis=1)
            
            # Filtrar clases con solo 1 muestra (rompen el split train/test)
            counts = df["stratify_col"].value_counts()
            valid_strata = counts[counts > 1].index
            initial_len = len(df)
            df = df[df["stratify_col"].isin(valid_strata)].reset_index(drop=True)
            
            if len(df) < initial_len:
                logger.info(f"Se eliminaron {initial_len - len(df)} filas por estratificación única.")
        else:
            logger.warning(f"Columnas de estratificación {s_cols} no encontradas. Usando default.")
            df["stratify_col"] = "default"
    else:
        df["stratify_col"] = "default"

    return df