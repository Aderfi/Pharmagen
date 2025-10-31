from pathlib import Path

import joblib
import pandas as pd
import torch
import numpy as np
from src.config.config import MODEL_ENCODERS_DIR
from sklearn.preprocessing import LabelEncoder

from .model_configs import MULTI_LABEL_COLUMN_NAMES

# <--- AÑADIDO: Token para etiquetas desconocidas (debe coincidir con PGenDataProcess)
UNKNOWN_TOKEN = "__UNKNOWN__"


def _transform_input(encoder: LabelEncoder, label: str, device: torch.device):
    """
    Transforma de forma segura una única etiqueta de string a un tensor,
    manejando etiquetas desconocidas.
    """
    # 1. Comprobar si la etiqueta es conocida por el encoder
    if label not in encoder.classes_:
        print(f"Advertencia: Etiqueta '{label}' desconocida. Usando '{UNKNOWN_TOKEN}'.")
        label_to_transform = UNKNOWN_TOKEN
    else:
        label_to_transform = label

    # 2. Transformar la etiqueta (que ahora es conocida o es __UNKNOWN__)
    try:
        idx = encoder.transform([label_to_transform])[0] # type: ignore
        return torch.tensor([idx], dtype=torch.long, device=device)
    except ValueError:
        # Fallback por si __UNKNOWN__ tampoco estuviera en el vocabulario
        print(f"Error: Ni '{label}' ni '{UNKNOWN_TOKEN}' están en el vocabulario del encoder.")
        # Intentar usar el índice 0 como un "desconocido" genérico
        return torch.tensor([0], dtype=torch.long, device=device)


def _transform_batch(encoder: LabelEncoder, series: pd.Series, unknown_token: str):
    """
    Transforma de forma segura un Series de pandas, manejando etiquetas desconocidas.
    """
    known_labels = set(encoder.classes_)
    
    # 1. Reemplazar cualquier etiqueta no conocida por el token UNKNOWN
    transformed_series = series.astype(str).apply(
        lambda x: x if x in known_labels else unknown_token
    )
    
    # 2. Transformar de forma segura
    return encoder.transform(transformed_series)


def load_encoders(model_name, encoders_dir=None):
    """
    Carga los encoders guardados para un modelo específico.
    """
    if encoders_dir is None:
        encoders_dir = Path(MODEL_ENCODERS_DIR)

    encoders_file = Path(encoders_dir, f"encoders_{model_name}.pkl")

    if not encoders_file.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de encoders en {encoders_file}"
        )

    try:
        encoders = joblib.load(encoders_file)
        print(f"Encoders cargados correctamente desde: {encoders_file}")
        
        # <--- AÑADIDO: Asegurarse de que __UNKNOWN__ existe en los encoders
        # (Si PGenDataProcess no se corrió, esto lo arregla)
        for col, encoder in encoders.items():
            if isinstance(encoder, LabelEncoder):
                if UNKNOWN_TOKEN not in encoder.classes_:
                    print(f"Advertencia: Añadiendo '{UNKNOWN_TOKEN}' al encoder '{col}' para predicción.")
                    # Añadir el token al vocabulario del encoder cargado
                    encoder.classes_ = np.append(encoder.classes_, UNKNOWN_TOKEN)
        
        return encoders
    except Exception as e:
        raise Exception(f"Error al cargar los encoders: {e}")


def predict_single_input(drug, genotype, gene, allele, model, encoders, target_cols):
    """
    Predicción para una sola entrada, compatible con DeepFM_PGenModel
    (salida de diccionario y manejo de multi-etiqueta).
    """
    if model is None or encoders is None or target_cols is None:
        raise ValueError("Se requieren modelo, encoders y target_cols")

    device = next(model.parameters()).device
    model.eval()

    # --- 1. Transformar Inputs (Corregido con manejo de UNKNOWN) ---
    try:
        drug_tensor = _transform_input(encoders["drug"], drug, device)
        gene_tensor = _transform_input(encoders["gene"], gene, device)
        allele_tensor = _transform_input(encoders["allele"], allele, device)
        
        # <--- CORRECCIÓN 1: Usar la clave correcta del encoder ---
        geno_tensor = _transform_input(
            encoders["variant/haplotypes"], genotype, device
        )
        
    except Exception as e:
        print(f"Error fatal al transformar inputs: {e}")
        return None

    # --- 2. Obtener Predicciones (Diccionario) ---
    with torch.no_grad():
        # <--- CORRECCIÓN 2: Orden de argumentos ---
        # El orden debe coincidir con model.py -> forward(self, drug, genotype, gene, allele)
        predictions_dict = model(drug_tensor, geno_tensor, gene_tensor, allele_tensor)

    # --- 3. Procesar Salidas Dinámicamente ---
    results = {}
    for col in target_cols:
        if col not in predictions_dict:
            print(f"Advertencia: El modelo no devolvió la salida para '{col}'.")
            continue

        logits = predictions_dict[col]  # Shape [1, N_CLASSES]

        if col in MULTI_LABEL_COLUMN_NAMES:
            # --- Lógica Multi-Etiqueta ---
            probs = torch.sigmoid(logits)
            predicted_vector = (
                (probs > 0.5).float().cpu().numpy()
            )  # Shape [1, N_CLASSES]

            decoded_labels = encoders[col].inverse_transform(predicted_vector)
            results[col] = list(decoded_labels[0])

        else:
            # --- Lógica Etiqueta-Única ---
            predicted_idx = torch.argmax(logits, dim=1).item()
            decoded_label = encoders[col].inverse_transform([predicted_idx])[0]
            
            # No mostrar la etiqueta __UNKNOWN__ al usuario
            if decoded_label == UNKNOWN_TOKEN:
                decoded_label = "Desconocido (Etiqueta no vista en entrenamiento)"
            
            results[col] = decoded_label

    return results


def predict_from_file(file_path, model, encoders, target_cols):
    """
    Predicción desde archivo, compatible con DeepFM_PGenModel
    (salida de diccionario y manejo de multi-etiqueta).
    """
    if model is None or encoders is None or target_cols is None:
        raise ValueError("Se requieren modelo, encoders y target_cols")

    try:
        df = pd.read_csv(file_path, sep=";", dtype=str)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return []

    device = next(model.parameters()).device
    model.eval()

    # --- 1. Transformar Inputs (Batch, Corregido con manejo de UNKNOWN) ---
    try:
        drug_tensor = torch.tensor(
            _transform_batch(encoders["drug"], df["drug"], UNKNOWN_TOKEN),
            dtype=torch.long, device=device
        )
        gene_tensor = torch.tensor(
            _transform_batch(encoders["gene"], df["gene"], UNKNOWN_TOKEN),
            dtype=torch.long, device=device
        )
        allele_tensor = torch.tensor(
            _transform_batch(encoders["allele"], df["allele"], UNKNOWN_TOKEN),
            dtype=torch.long, device=device
        )
        
        # <--- CORRECCIÓN 1: Usar la clave correcta del encoder y la columna correcta ---
        # Transforma la columna "genotype" del CSV usando el encoder "variant/haplotypes"
        geno_tensor = torch.tensor(
            _transform_batch(encoders["variant/haplotypes"], df["genotype"], UNKNOWN_TOKEN),
            dtype=torch.long, device=device
        )
        
    except KeyError as e:
        print(f"Error: La columna {e} no se encontró en el CSV o el encoder no existe.")
        return []
    except Exception as e:
        print(f"Error al transformar inputs del archivo: {e}")
        return []

    # --- 2. Obtener Predicciones (Diccionario) ---
    with torch.no_grad():
        # <--- CORRECCIÓN 2: Orden de argumentos ---
        # El orden debe coincidir con model.py -> forward(self, drug, genotype, gene, allele)
        predictions_dict = model(drug_tensor, geno_tensor, gene_tensor, allele_tensor)

    # --- 3. Procesar Salidas (Batch) ---
    decoded_outputs = {}

    for col in target_cols:
        if col not in predictions_dict:
            continue

        logits = predictions_dict[col]  # Shape [N_BATCH, N_CLASSES]

        if col in MULTI_LABEL_COLUMN_NAMES:
            # --- Lógica Multi-Etiqueta ---
            probs = torch.sigmoid(logits)
            predicted_matrix = (
                (probs > 0.5).float().cpu().numpy()
            )  # Shape [N_BATCH, N_CLASSES]
            
            decoded_labels_tuples = encoders[col].inverse_transform(predicted_matrix)
            decoded_outputs[col] = [list(labels) for labels in decoded_labels_tuples]

        else:
            # --- Lógica Etiqueta-Única ---
            predicted_indices = torch.argmax(logits, dim=1).cpu().numpy()
            decoded_labels = encoders[col].inverse_transform(predicted_indices)
            
            # No mostrar la etiqueta __UNKNOWN__ al usuario
            decoded_outputs[col] = [
                "Desconocido" if label == UNKNOWN_TOKEN else label
                for label in decoded_labels
            ]

    # --- 4. Re-ensamblar en la Estructura de Salida ---
    results = []
    num_rows = len(df)

    for i in range(num_rows):
        row_result = {
            # Añadir inputs originales
            "drug": df.iloc[i]["drug"],
            "gene": df.iloc[i]["gene"],
            "allele": df.iloc[i]["allele"],
            "genotype": df.iloc[i]["variant/haplotypes"],
            # Añadir predicciones
            **{
                col: decoded_outputs[col][i]
                for col in target_cols
                if col in decoded_outputs
            },
        }
        results.append(row_result)

    return results