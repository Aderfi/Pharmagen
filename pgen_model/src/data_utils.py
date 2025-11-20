import logging
import re
import json
from pathlib import Path
from typing import List, Optional, Union, Any, Set

import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz

from src.config.config import DATA_DIR

logger = logging.getLogger(__name__)
UNKNOWN_TOKEN = "__UNKNOWN__"

def serialize_multi_labelcols(label_input: Any) -> str:
    """
    Serializa entradas (listas, arrays, strings) a un string ordenado separado por pipes.
    """
    if pd.isna(label_input) or label_input in ["nan", "", "null", None]:
        return ""
    
    parts: Set[str] = set()
    
    if isinstance(label_input, str):
        # Divide por comas, puntos y coma o pipes existentes
        parts = {s.strip() for s in re.split(r"[,;|]+", label_input) if s.strip()}
    elif isinstance(label_input, (list, tuple, np.ndarray, set)):
        parts = {str(x).strip() for x in label_input if x is not None}
        
    return "|".join(sorted(parts))

def drugs_to_atc(df: pd.DataFrame, drug_col: str, atc_col: str = "atc", lang_idx: int = 1) -> pd.DataFrame:
    """
    Mapea nombres de fármacos a códigos ATC con caché y búsqueda fuzzy optimizada.
    """
    json_path = Path(DATA_DIR) / "dicts" / "ATC_drug_med.json"
    
    if not json_path.exists():
        logger.warning(f"Diccionario ATC no encontrado en {json_path}. Saltando mapeo.")
        df[atc_col] = "No_ATC"
        return df

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            atc_dict = json.load(f)
    except Exception as e:
        logger.error(f"Error leyendo JSON ATC: {e}")
        return df

    # Mapa inverso optimizado (Nombre -> Código)
    atc_dict_rev = {v[lang_idx].lower(): k for k, v in atc_dict.items()}
    atc_keys = list(atc_dict_rev.keys())
    
    # Cache local para evitar re-calcular fuzzy matches repetidos
    memoization = {}

    def get_atc_codes(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "No_ATC"
        
        text_lower = text.lower().strip()
        if text_lower in memoization:
            return memoization[text_lower]

        tokens = [p.strip() for p in re.split(r'[;,/|]+', text_lower) if p.strip()]
        codes = []

        for token in tokens:
            if token in atc_dict_rev:
                codes.append(atc_dict_rev[token])
            else:
                # Fuzzy match estricto (score > 90)
                result = process.extractOne(token, atc_keys, scorer=fuzz.WRatio, score_cutoff=90)
                if result:
                    match_name = result[0]
                    codes.append(atc_dict_rev[match_name])
                else:
                    codes.append("No_ATC")
        
        final_code = "|".join(sorted(set(codes))) if codes else "No_ATC"
        memoization[text_lower] = final_code
        return final_code

    logger.info("Mapeando fármacos a ATC...")
    # map() es más rápido que apply() para operaciones elemento a elemento
    df[atc_col] = df[drug_col].map(get_atc_codes).astype("string")
    return df

def data_import_normalize(
    df: pd.DataFrame, 
    all_cols: List[str], 
    target_cols: List[str], 
    multi_label_targets: Optional[List[str]], 
    stratify_cols: Union[List[str], str]
) -> pd.DataFrame:
    
    multi_label_targets = multi_label_targets or []
    
    # 1. Limpieza Vectorizada de Single Label y Features
    single_label_cols = [c for c in all_cols if c in df.columns and c not in multi_label_targets]
    
    if single_label_cols:
        df[single_label_cols] = df[single_label_cols].fillna("UNKNOWN").astype(str)
        for col in single_label_cols:
            # Reemplazar separadores y espacios
             df[col] = df[col].str.replace(r'[,;]+', '|', regex=True).str.replace(' ', '_', regex=False).str.strip().str.lower()

    # 2. Limpieza Multi Label
    for col in multi_label_targets:
        if col in df.columns:
            df[col] = df[col].apply(serialize_multi_labelcols).str.lower()

    # 3. Creación de columna de estratificación
    s_cols = [stratify_cols] if isinstance(stratify_cols, str) else stratify_cols
    existing_strat_cols = [c for c in s_cols if c in df.columns]
    
    if existing_strat_cols:
        df['stratify_col'] = df[existing_strat_cols].astype(str).agg('|'.join, axis=1)
    else:
        df['stratify_col'] = "default"

    # Normalizar nombres de columnas (minúsculas y sin espacios)
    df.columns = df.columns.str.lower().str.strip()
    
    return df