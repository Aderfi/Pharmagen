"""
Módulo de predicción para Pharmagen PModel.

Incluye funciones para:
- Predecir a partir de una única entrada manual (mutaciones y medicamentos).
- Predecir a partir de un archivo CSV con múltiples pacientes.
"""

import torch
from .model import PGenModel

def load_trained_model(model_path, params, device=None):
    """
    Carga un modelo previamente entrenado desde disco.

    Args:
        model_path (str): Ruta al archivo .pth con los pesos del modelo.
        params (dict): Diccionario con las dimensiones y parámetros para crear la arquitectura.
            Debe incluir las claves:
                - n_drug_genos
                - emb_dim
                - hidden_dim
                - dropout_rate
        device (str o torch.device, opcional): dispositivo ("cpu" o "cuda").

    Returns:
        model (PGenModel): Modelo cargado y puesto en modo evaluación.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instanciar el modelo con la arquitectura adecuada
    model = PGenModel(
        params['n_drug_genos'],
        params['emb_dim'],
        params['hidden_dim'],
        params['dropout_rate'],
        params['output_dims']
    ).to(device)

    # Cargar los pesos
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_single_input(mutaciones, medicamentos, model=None, encoders=None):
    """
    Realiza una predicción a partir de los datos introducidos manualmente.
    
    mutaciones: str (ejemplo: "TP53,BRCA1")
    medicamentos: str (ejemplo: "cisplatino,5FU")
    model: instancia de PGenModel entrenada (opcional)
    encoders: diccionario de LabelEncoders (opcional)
    """
    # Preprocesar la entrada y formar la combinación
    drug_geno = medicamentos.strip() + "_" + mutaciones.strip()
    if encoders is None or 'Drug_Geno' not in encoders:
        raise ValueError("Se requiere un encoder para la columna Drug_Geno")
    idx = encoders['Drug_Geno'].transform([drug_geno])[0]
    input_tensor = torch.tensor([idx], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
    # Decodificar predicción
    result = {}
    for key, logits in outputs.items():
        pred_idx = torch.argmax(logits, dim=1).item()
        if key.capitalize() in encoders:  # encoders de targets
            result[key.capitalize()] = encoders[key.capitalize()].inverse_transform([pred_idx])[0]
        else:
            result[key.capitalize()] = pred_idx
    return result

def predict_from_file(file_path, model=None, encoders=None):
    """
    Realiza predicciones para varios pacientes a partir de un archivo CSV.
    
    file_path: ruta al archivo CSV de entrada.
    model: instancia de PGenModel entrenada (opcional)
    encoders: diccionario de LabelEncoders (opcional)
    """
    import pandas as pd
    df = pd.read_csv(file_path, sep=';', dtype=str)
    df["Drug_Geno"] = df["Drug"].astype(str) + "_" + df["Genotype"].astype(str)
    idxs = encoders['Drug_Geno'].transform(df["Drug_Geno"].tolist())
    input_tensor = torch.tensor(idxs, dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_tensor)
    results = []
    for i in range(len(df)):
        row_result = {}
        for key, logits in outputs.items():
            pred_idx = torch.argmax(logits[i]).item()
            if key.capitalize() in encoders:
                row_result[key.capitalize()] = encoders[key.capitalize()].inverse_transform([pred_idx])[0]
            else:
                row_result[key.capitalize()] = pred_idx
        results.append(row_result)
    return results
