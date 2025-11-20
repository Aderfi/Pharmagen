from pathlib import Path
from typing import Dict, List, Any, Union

import joblib
import pandas as pd
import torch
import numpy as np
from src.config.config import MODEL_ENCODERS_DIR
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

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
    Optimizado con numpy.isin.
    """
    # 1. Convertir a numpy array de strings
    vals = series.astype(str).to_numpy()
    
    # 2. Identificar valores desconocidos
    mask = ~np.isin(vals, encoder.classes_)
    
    # 3. Reemplazar desconocidos con el token
    if mask.any():
        vals[mask] = unknown_token
        
    # 4. Transformar (ahora seguro)
    return encoder.transform(vals)


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
        encoders: dict[str, LabelEncoder | MultiLabelBinarizer] = joblib.load(encoders_file)
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


def predict_single_input(features_dict, model, encoders: Dict[str, Any], target_cols):
    """
    Predicción para una sola entrada, compatible con DeepFM_PGenModel
    (salida de diccionario y manejo de multi-etiqueta).
    """
    if model is None or encoders is None or target_cols is None:
        raise ValueError("Se requieren modelo, encoders y target_cols")

    device = next(model.parameters()).device
    model.eval()

    model_inputs = {}
    
    # --- 1. Transformar Inputs ---
    try:
        # Iteramos sobre las características proporcionadas en el diccionario
        for feature_name, feature_value in features_dict.items():
            
            # Verificamos si tenemos un encoder para esta característica
            if feature_name in encoders:
                # Asumimos que _transform_input es tu función auxiliar existente
                tensor = _transform_input(encoders[feature_name], feature_value, device)
                model_inputs[feature_name] = tensor
            else:
                # Opcional: Loggear si llega un feature que no esperábamos o no tiene encoder
                print(f"Ignorando feature '{feature_name}' por falta de encoder.")
                pass
    except Exception as e:
        print(f"Error fatal al transformar inputs: {e}")
        return None

    # --- 2. Obtener Predicciones (Diccionario) ---
    with torch.no_grad():
        try:
            predictions_dict = model(**model_inputs)
        except TypeError as e:
            print(f"Error de argumentos en el modelo: {e}")
            print(f"Inputs enviados: {list(model_inputs.keys())}")
            return None

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
                decoded_label = " --- "
            
            results[col] = decoded_label

    return results


def predict_from_file(file_path, model, encoders, target_cols, batch_size=1024):
    """
    Predicción desde archivo optimizada (vectorizada).
    Procesa todo el archivo en batches para evitar OOM en archivos muy grandes,
    pero usa operaciones vectorizadas dentro de cada batch.
    """
    if model is None or encoders is None or target_cols is None:
        raise ValueError("Se requieren modelo, encoders y target_cols")

    ext = Path(file_path).suffix.lower()
    ext_sep = {
            ".csv": ",", 
            ".tsv": "\t"
    }.get(ext, ",")
    
    try:
        # Leer todo el DF (asumiendo que cabe en memoria, si es gigante usar chunks)
        df = pd.read_csv(file_path, sep=ext_sep, dtype=str)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return []

    device = next(model.parameters()).device
    model.eval()
    
    # Preparar columnas de entrada
    # Mapeo de nombres de columnas del CSV a nombres de encoders/modelo
    col_map = {
        "drug": "drug",
        "gene": "gene",
        "allele": "allele",
        "genotype": "variant/haplotypes" # CSV tiene 'genotype', encoder tiene 'variant/haplotypes'
    }
    
    # Verificar columnas
    for csv_col in col_map.keys():
        if csv_col not in df.columns:
             print(f"Error: La columna '{csv_col}' no se encontró en el archivo.")
             return []

    # --- 1. Transformar Inputs (Vectorizado) ---
    # Se transforman todas las filas de una vez (es rápido con numpy)
    input_tensors = {}
    try:
        for csv_col, encoder_key in col_map.items():
            if encoder_key not in encoders:
                print(f"Error: Encoder '{encoder_key}' no encontrado.")
                return []
            
            encoded_vals = _transform_batch(encoders[encoder_key], df[csv_col], UNKNOWN_TOKEN)
            # Guardar como tensor en CPU, moveremos a GPU por batches
            input_tensors[csv_col] = torch.tensor(encoded_vals, dtype=torch.long)
            
    except Exception as e:
        print(f"Error al transformar inputs del archivo: {e}")
        return []

    # --- 2. Inferencia por Batches ---
    num_samples = len(df)
    all_predictions = {col: [] for col in target_cols}
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            # Slice del batch
            batch_end = min(i + batch_size, num_samples)
            
            # Preparar inputs del batch y mover a device
            # El orden de argumentos en forward es importante si no se usan kwargs
            # DeepFM_PGenModel.forward(drug, genalle, gene, allele) -> Nombres en modelo
            # Mapeo: drug->drug, genotype->genalle, gene->gene, allele->allele
            
            b_drug = input_tensors["drug"][i:batch_end].to(device)
            b_genalle = input_tensors["genotype"][i:batch_end].to(device)
            b_gene = input_tensors["gene"][i:batch_end].to(device)
            b_allele = input_tensors["allele"][i:batch_end].to(device)
            
            # Forward
            # IMPORTANTE: El modelo espera argumentos posicionales o kwargs que coincidan
            # forward(self, inputs: Dict) o forward(self, drug, genalle, gene, allele)
            # Revisando model.py, forward toma un dict 'inputs' O kwargs.
            # Pero en kge_test.py y otros sitios se llama con args posicionales?
            # En model.py original: forward(self, inputs: Dict[str, torch.Tensor], **kwargs)
            # Así que pasamos kwargs.
            
            batch_outputs = model(
                drug=b_drug, 
                genalle=b_genalle, 
                gene=b_gene, 
                allele=b_allele
            )
            
            # Procesar outputs del batch
            for col in target_cols:
                if col not in batch_outputs:
                    continue
                    
                logits = batch_outputs[col]
                
                if col in MULTI_LABEL_COLUMN_NAMES:
                    # Multi-label: Sigmoid > 0.5
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float().cpu().numpy()
                    all_predictions[col].append(preds)
                else:
                    # Single-label: Argmax
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    all_predictions[col].append(preds)

    # --- 3. Decodificar y Ensamblar (Vectorizado) ---
    # Concatenar resultados de batches
    final_results = df.copy()
    
    for col in target_cols:
        if not all_predictions[col]:
            continue
            
        # Concatenar lista de arrays
        preds_array = np.concatenate(all_predictions[col], axis=0)
        
        if col in MULTI_LABEL_COLUMN_NAMES:
            # Inverse transform de multilabel es lento (lista de tuplas)
            # No hay forma vectorizada directa en sklearn para inverse_transform de MLB que devuelva strings planos
            decoded = encoders[col].inverse_transform(preds_array)
            # Convertir tuplas a listas
            final_results[col] = [list(d) for d in decoded]
        else:
            # Inverse transform de label encoder es rápido
            decoded = encoders[col].inverse_transform(preds_array)
            
            # Reemplazar UNKNOWN_TOKEN con "Desconocido" (vectorizado)
            # Usamos numpy para reemplazo rápido
            decoded = np.where(decoded == UNKNOWN_TOKEN, "Desconocido", decoded)
            final_results[col] = decoded

    # Convertir a lista de dicts para mantener compatibilidad de retorno
    return final_results.to_dict(orient="records")