# Pharmagen - Data Utilities
import re
import pandas as pd
import numpy as np
from typing import Any
from collections.abc import Iterable


RE_SPLITTERS = re.compile(r"[,;|]+")

def clean_dataset_content(
    df: pd.DataFrame,
    multi_label_cols: Iterable[str] | None = None,
    unknown_token: str = "__UNKNOWN__"
) -> pd.DataFrame:
    """
    Performs vectorized cleaning of the DataFrame content:
    1. Normalizes headers (lowercase, strip).
    2. Fills NaNs with unknown_token.
    3. Converts all cells to lowercase string.
    4. Normalizes multi-label delimiters to pipes '|'.

    Args:
        df: Input DataFrame.
        multi_label_cols: List of column names that contain multi-label data.
        unknown_token: Token to use for missing values.

    Returns:
        Cleaned DataFrame.
    """
    # 1. Normalize headers
    df.columns = df.columns.str.lower().str.strip()

    # 2. Vectorized String Cleaning
    delimiters_re = re.compile(r"[,;]+")
    
    # Prepare set for O(1) lookup
    multi_label_set = set(c.lower() for c in (multi_label_cols or []))

    for col in df.columns:
        # Fast conversion to string and lowercasing
        series = df[col].fillna(unknown_token).astype(str).str.strip().str.lower()

        if col in multi_label_set:
            # Normalize delimiters to pipe '|' for multi-label consistency
            df[col] = series.apply(lambda x: delimiters_re.sub("|", x))
        else:
            df[col] = series
            
    return df

def serialize_multilabel(val: Any) -> str:
    """
    Standardizes multi-label strings to pipe-separated format.
    e.g. "A, B; C" -> "A|B|C"
    """
    if pd.isna(val) or val == "":
        return ""

    if isinstance(val, str):
        parts = {s.strip() for s in RE_SPLITTERS.split(val) if s.strip()}
    elif isinstance(val, (list, tuple, np.ndarray)):
        parts = {str(x).strip() for x in val if x}
    else:
        return str(val)
        
    return "|".join(sorted(parts))
