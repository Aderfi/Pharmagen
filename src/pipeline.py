# coding=utf-8
"""
Pipeline de entrenamiento para modelos farmacogenéticos.

Gestiona el flujo completo de entrenamiento incluyendo:
- Carga y preprocesamiento de datos
- División train/validation
- Configuración de hiperparámetros
- Entrenamiento del modelo
- Guardado de modelos y encoders
"""

import logging
import multiprocessing
import os
import random
from pathlib import Path
from typing import List, Any

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.config.config import MODEL_ENCODERS_DIR, MODELS_DIR, MODEL_TRAIN_DATA
from torch.utils.data import DataLoader

from .data import PGenDataset, PGenDataProcess
from .model import DeepFM_PGenModel
from .model_configs import MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model, save_model
from .train_utils import (
    create_optimizer,
    create_scheduler,
    create_criterions,
)  # Import from train_utils

logger = logging.getLogger(__name__)


def train_pipeline(
    PGEN_MODEL_DIR,
    csv_path: Path,  # Changed to Path and single csv_path
    model_name,
    target_cols=None,
    patience: int = 25,  # Use constant from config or a default
    epochs: int = 100,  # Use constant from config or a default
):
    """
    Función principal para entrenar un modelo PGen.

    Args:
        PGEN_MODEL_DIR (str): Directorio base para guardar modelos y resultados.
        csv_path (Path): Ruta al archivo CSV/TSV con datos de entrenamiento.
        model_name (str): Nombre del modelo a entrenar (debe coincidir con una configuración).
        target_cols (list, optional): Lista de columnas target a predecir. Si es None, se usan las definidas en la configuración del modelo.
        patience (int, optional): Número de épocas sin mejora antes de detener el entrenamiento temprano.
        epochs (int, optional): Número máximo de épocas para entrenar.

    Returns:
        None
    """
    seed = 711
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configuración
    config = get_model_config(model_name)
    feature_cols = [c.lower() for c in config["features"]]
    target_cols = [t.lower() for t in config["targets"]]

    # Datos
    processor = PGenDataProcess()
    df = processor.load_data(
        csv_path=csv_path,  # Use the renamed parameter
        all_cols=config["cols"],
        input_cols=feature_cols,
        target_cols=target_cols,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=config["stratify"],
    )

    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["stratify_col"], random_state=seed
    )

    processor.fit(train_df)
    train_enc = processor.transform(train_df)
    val_enc = processor.transform(val_df)

    train_ds = PGenDataset(
        train_enc, feature_cols, target_cols, MULTI_LABEL_COLUMN_NAMES
    )
    val_ds = PGenDataset(val_enc, feature_cols, target_cols, MULTI_LABEL_COLUMN_NAMES)

    train_loader = DataLoader(
        train_ds,
        batch_size=config["params"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["params"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Dimensiones
    n_features = {}
    target_dims = {}
    for col, enc in processor.encoders.items():
        # Only include features/targets that are actually used by the model based on config
        if col in feature_cols:
            n_features[col] = len(enc.classes_)
        if col in target_cols:
            target_dims[col] = len(enc.classes_)

    # Modelo
    params = config["params"]
    model = DeepFM_PGenModel(
        n_features=n_features,
        target_dims=target_dims,
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        n_layers=params["n_layers"],
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.0),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 64),
        embedding_dropout=params.get("embedding_dropout", 0.0),
    ).to(device)

    # Setup
    criterions: List[Any] = create_criterions(target_cols, params, device)
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)
    criterions.append(optimizer)

    # Entrenar
    print(f"Entrenando {model_name}...")
    best_loss, best_acc, task_losses = train_model(
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=epochs,
        patience=patience,
        model_name=model_name,  # Use pipeline's patience/epochs
        feature_cols=feature_cols,
        target_cols=target_cols,
        device=device,
        scheduler=scheduler,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        return_per_task_losses=True,
        progress_bar=True,
    )

    # Guardar
    save_model(model, target_cols, best_loss, best_acc, model_name, task_losses, params)

    # Guardar Encoders
    enc_path = Path(MODEL_ENCODERS_DIR) / f"encoders_{model_name}.pkl"
    joblib.dump(processor.encoders, enc_path)
    print(f"Encoders guardados en {enc_path}")
