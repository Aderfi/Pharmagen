# Pharmagen - Data Utilities
import re
import pandas as pd
import numpy as np
from typing import Any, cast


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

def normalize_drug_name(name):
    import pubchempy as pcp
    from pubchempy import Compound
    
    try:
        # Busca el compuesto por nombre
        compounds: list[Compound] = cast(list[Compound], pcp.get_compounds(name, 'error'))
        
        if compounds:
            # Devuelve el primer resultado (normalmente el más relevante)
            c = compounds[0]
            return {
                'Input': name,
                'Synonyms': c.synonyms[:3], # Primeros 3 sinónimos
                'IUPAC': c.iupac_name,
                'SMILES': c.isomeric_smiles, # Útil para RDKit
                'CID': c.cid
            }
    except Exception as e:
        return None

if __name__ == "__main__":

    drug = "amphetamine"

    norm_drug = normalize_drug_name(drug)

    test = norm_drug

    print(test)

