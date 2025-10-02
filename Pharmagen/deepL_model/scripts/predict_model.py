import sys
import pandas as pd
from Pharmagen.config import MODELS_DIR, MODEL_VOCABS_DIR, PROCESSED_DATA_DIR
from Pharmagen.src.data_handle.preprocess_model import PharmagenPreprocessor
from Pharmagen.deepL_model.scripts.model_utils import load_pharmagen_model
import os

# 1. Configuración de rutas (centralizadas en config.py)
MODEL_PATH = MODELS_DIR / "modelo_pharmagen_final.h5"
MUT_VOCAB_PATH = MODEL_VOCABS_DIR / "mut_vocab.json"
DRUG_VOCAB_PATH = MODEL_VOCABS_DIR / "drug_vocab.json"

def predict_single_input(mutaciones: str, medicamentos: str) -> dict:
    """Predice para una entrada manual."""
    model, preproc = load_pharmagen_model(str(MODEL_PATH), str(MUT_VOCAB_PATH), str(DRUG_VOCAB_PATH))
    if model is None:
        raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}")
    if preproc is None:
        raise RuntimeError("No se pudo cargar el preprocesador o los vocabularios.")
    df = pd.DataFrame({"mutaciones": [mutaciones], "medicamentos": [medicamentos]})
    X_mut, X_drug = preproc.transform(df, "mutaciones", "medicamentos")
    if hasattr(model, "predict"):
        pred = model.predict([X_mut, X_drug])[0][0]
    else:
        raise AttributeError("El objeto model no tiene el método predict.")
    return {
        "Mutaciones": mutaciones,
        "Medicamentos": medicamentos,
        "Predicción (score)": float(pred),
        "Clase": int(pred > 0.5)
    }

def predict_from_file(file_path: str) -> pd.DataFrame:
    """Predice para un archivo CSV con columnas mutaciones y medicamentos."""
    model, preproc = load_pharmagen_model(str(MODEL_PATH), str(MUT_VOCAB_PATH), str(DRUG_VOCAB_PATH))
    if model is None:
        raise RuntimeError(f"No se pudo cargar el modelo desde {MODEL_PATH}")
    if preproc is None:
        raise RuntimeError("No se pudo cargar el preprocesador o los vocabularios.")
    df = pd.read_csv(file_path)
    if not {"mutaciones", "medicamentos"}.issubset(df.columns):
        raise ValueError("El archivo debe contener columnas 'mutaciones' y 'medicamentos'.")
    X_mut, X_drug = preproc.transform(df, "mutaciones", "medicamentos")
    if hasattr(model, "predict"):
        preds = model.predict([X_mut, X_drug]).flatten()
    else:
        raise AttributeError("El objeto model no tiene el método predict.")
    df["Predicción (score)"] = preds
    df["Clase"] = (preds > 0.5).astype(int)
    out_path = os.path.splitext(file_path)[0] + "_predicciones.csv"
    df.to_csv(out_path, index=False)
    print(f"Resultados guardados en: {out_path}")
    return df[["mutaciones", "medicamentos", "Predicción (score)", "Clase"]]

def main():
    print("Este script no debe ejecutarse directamente. Utilízalo a través de main.py.")

if __name__ == "__main__":
    main()