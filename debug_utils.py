import joblib

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import numpy as np
from pathlib import Path
from src.cfg.config import MODEL_ENCODERS_DIR

file_name = "f"
ENCODERS_FILE = Path(MODEL_ENCODERS_DIR / file_name)
UNKNOWN_TOKEN = "__UNKNOWN__"

def inspect_encoders(encoders_file = ENCODERS_FILE):
    print(f"--- Inspección de Encoders ---")
    encoders: dict[str, LabelEncoder | MultiLabelBinarizer] = joblib.load(encoders_file)
    print(f"Encoders cargados correctamente desde: {encoders_file}")
        
    for col, encoder in encoders.items():
        if isinstance(encoder, LabelEncoder):
            if UNKNOWN_TOKEN not in encoder.classes_:
                print(f"Advertencia: Añadiendo '{UNKNOWN_TOKEN}' al encoder '{col}' para predicción.")
                encoder.classes_ = np.append(encoder.classes_, UNKNOWN_TOKEN)
    
    for name, obj in encoders.items():
        print(f"\nFeature: '{name}'")
        print(f"  Tipo: {type(obj).__name__}")
        
        if hasattr(obj, 'classes_'):
            print(f"  Nº Clases: {len(obj.classes_)}")
            print(f"  Ejemplos: {obj.classes_[:3]}...") # Muestra los primeros 3
        else:
            print("  Advertencia: No tiene atributo classes_")