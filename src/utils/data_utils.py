# Pharmagen - Data Utilities
import re
import pandas as pd
import numpy as np
from typing import Any

RE_SPLITTERS = re.compile(r"[,;|]+")

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
