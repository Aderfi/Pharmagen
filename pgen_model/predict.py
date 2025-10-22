import joblib
import pandas as pd
import torch

from pathlib import Path

from .model_configs import MULTI_LABEL_COLUMN_NAMES
from src.config.config import MODEL_ENCODERS_DIR


def load_encoders(model_name, encoders_dir=None):
    """
    Carga los encoders guardados para un modelo específico.
    
    Args:
        model_name (str): Nombre del modelo cuyos encoders quieres cargar
        base_dir (str, optional): Directorio base donde buscar. Si es None,
                                  usa el directorio de modelos por defecto.
    
    Returns:
        dict: Diccionario de encoders (LabelEncoder y MultiLabelBinarizer)
    """
    if encoders_dir is None:
        encoders_dir = Path(MODEL_ENCODERS_DIR)  
    
    encoders_file = Path(encoders_dir, f'encoders_{model_name}.pkl')
    
    if not encoders_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de encoders en {encoders_file}")
    
    try:
        encoders = joblib.load(encoders_file)
        print(f"Encoders cargados correctamente desde: {encoders_file}")
        return encoders
    except Exception as e:
        raise Exception(f"Error al cargar los encoders: {e}")

def predict_single_input(drug, gene, allele, genotype, model, encoders, target_cols):
    """
    Predicción para una sola entrada, compatible con DeepFM_PGenModel
    (salida de diccionario y manejo de multi-etiqueta).
    """
    if model is None or encoders is None or target_cols is None:
        raise ValueError("Se requieren modelo, encoders y target_cols")

    device = next(model.parameters()).device
    model.eval()

    # --- 1. Transformar Inputs ---
    try:
        drug_tensor = torch.tensor([encoders['drug'].transform([drug])[0]], dtype=torch.long, device=device)
        gene_tensor = torch.tensor([encoders['gene'].transform([gene])[0]], dtype=torch.long, device=device)
        allele_tensor = torch.tensor([encoders['allele'].transform([allele])[0]], dtype=torch.long, device=device)
        geno_tensor = torch.tensor([encoders['genotype'].transform([genotype])[0]], dtype=torch.long, device=device)
    except Exception as e:
        print(f"Error al transformar inputs. ¿Están las categorías en los encoders? {e}")
        return None

    # --- 2. Obtener Predicciones (Diccionario) ---
    with torch.no_grad():
        predictions_dict = model(drug_tensor, gene_tensor, allele_tensor, geno_tensor)

    # --- 3. Procesar Salidas Dinámicamente ---
    results = {}
    for col in target_cols:
        if col not in predictions_dict:
            print(f"Advertencia: El modelo no devolvió la salida para '{col}'.")
            continue
        
        logits = predictions_dict[col] # Shape [1, N_CLASSES]
        
        if col in MULTI_LABEL_COLUMN_NAMES:
            # --- Lógica Multi-Etiqueta ---
            probs = torch.sigmoid(logits)
            # Vector binario de predicción, ej. [0., 1., 0., 1.]
            predicted_vector = (probs > 0.5).float().cpu().numpy() # Shape [1, N_CLASSES]
            
            # inverse_transform de MultiLabelBinarizer espera una matriz 2D
            decoded_labels = encoders[col].inverse_transform(predicted_vector)
            # `decoded_labels` es una tupla de tuplas, ej: (('LabelA', 'LabelD'),)
            # Lo convertimos a una lista simple de strings
            results[col] = list(decoded_labels[0])

        else:
            # --- Lógica Etiqueta-Única ---
            predicted_idx = torch.argmax(logits, dim=1).item()
            # inverse_transform de LabelEncoder espera una lista de índices
            decoded_label = encoders[col].inverse_transform([predicted_idx])[0]
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
        df = pd.read_csv(file_path, sep=';', dtype=str)
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return []

    device = next(model.parameters()).device
    model.eval()

    # --- 1. Transformar Inputs (Batch) ---
    try:
        drug_tensor = torch.tensor(encoders['drug'].transform(df["drug"].astype(str)), dtype=torch.long, device=device)
        gene_tensor = torch.tensor(encoders['gene'].transform(df["gene"].astype(str)), dtype=torch.long, device=device)
        allele_tensor = torch.tensor(encoders['allele'].transform(df["allele"].astype(str)), dtype=torch.long, device=device)
        geno_tensor = torch.tensor(encoders['genotype'].transform(df["genotype"].astype(str)), dtype=torch.long, device=device)
    except Exception as e:
        print(f"Error al transformar inputs del archivo. ¿Categorías desconocidas? {e}")
        return []

    # --- 2. Obtener Predicciones (Diccionario) ---
    with torch.no_grad():
        predictions_dict = model(drug_tensor, gene_tensor, allele_tensor, geno_tensor)

    # --- 3. Procesar Salidas (Batch) ---
    # Almacenaremos las predicciones decodificadas para cada columna
    decoded_outputs = {}
    
    for col in target_cols:
        if col not in predictions_dict:
            continue
        
        logits = predictions_dict[col] # Shape [N_BATCH, N_CLASSES]
        
        if col in MULTI_LABEL_COLUMN_NAMES:
            # --- Lógica Multi-Etiqueta ---
            probs = torch.sigmoid(logits)
            predicted_matrix = (probs > 0.5).float().cpu().numpy() # Shape [N_BATCH, N_CLASSES]
            
            # inverse_transform decodifica la matriz entera de una vez
            # Devuelve una lista de tuplas, ej: [('LabelA',), ('LabelB', 'LabelC'), ...]
            decoded_labels_tuples = encoders[col].inverse_transform(predicted_matrix)
            # Convertimos a listas de strings
            decoded_outputs[col] = [list(labels) for labels in decoded_labels_tuples]

        else:
            # --- Lógica Etiqueta-Única ---
            predicted_indices = torch.argmax(logits, dim=1).cpu().numpy()
            # inverse_transform decodifica el array entero de índices
            decoded_outputs[col] = encoders[col].inverse_transform(predicted_indices)

    # --- 4. Re-ensamblar en la Estructura de Salida ---
    results = []
    num_rows = len(df)
    
    for i in range(num_rows):
        row_result = {
            # Añadir inputs originales
            "drug": df.iloc[i]["drug"],
            "gene": df.iloc[i]["gene"],
            "allele": df.iloc[i]["allele"],
            "genotype": df.iloc[i]["genotype"],
            # Añadir predicciones
            **{col: decoded_outputs[col][i] for col in target_cols if col in decoded_outputs}
        }
        results.append(row_result)

    return results