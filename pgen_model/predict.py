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
                - n_drugs
                - n_genotypes
                - n_outcomes
                - n_variations
                - n_effects
                - n_entities
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
        params['n_drugs'],
        params['n_genotypes'],
        params['n_outcomes'],
        params['n_variations'],
        params['n_effects'],
        params['n_entities'],
        params['emb_dim'],
        params['hidden_dim'],
        params['dropout_rate']
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
    # TODO: Preprocesar la entrada, codificar usando los encoders, preparar tensores
    # TODO: Ejecutar el modelo y decodificar la predicción
    # Ejemplo de retorno: {"Outcome": "Sensitive", "Effect": "High", ...}
    pass

def predict_from_file(file_path, model=None, encoders=None):
    """
    Realiza predicciones para varios pacientes a partir de un archivo CSV.
    
    file_path: ruta al archivo CSV de entrada.
    model: instancia de PGenModel entrenada (opcional)
    encoders: diccionario de LabelEncoders (opcional)
    """
    # TODO: Leer el archivo, procesar cada fila, codificar, preparar batch
    # TODO: Ejecutar el modelo y decodificar las predicciones
    # Ejemplo de retorno: DataFrame o lista de dicts con resultados para cada paciente
    pass