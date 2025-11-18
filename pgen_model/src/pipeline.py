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

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from src.config.config import MODEL_ENCODERS_DIR, MODELS_DIR
from torch.utils.data import DataLoader

from .data import PGenDataset, PGenDataProcess
from .model import DeepFM_PGenModel
from .model_configs import CLINICAL_PRIORITIES, MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model, save_model
from .train_utils import create_optimizer, create_scheduler, create_criterions
from .focal_loss import FocalLoss

logger = logging.getLogger(__name__)

# Constantes de configuración por defecto
EPOCHS = 100
PATIENCE = 25


def train_pipeline(
    PGEN_MODEL_DIR,
    csv_files, # Este argumento parece no usarse, pero lo mantenemos
    model_name,
    target_cols=None,
    patience=PATIENCE,
    epochs=EPOCHS,
):
    """
    Función principal para entrenar un modelo PGen.
    
    Args: 
        PGEN_MODEL_DIR (str): Directorio base para guardar modelos y resultados.
        csv_files (list): Lista de rutas a archivos CSV/TSV con datos de entrenamiento.
        model_name (str): Nombre del modelo a entrenar (debe coincidir con una configuración).
        target_cols (list, optional): Lista de columnas target a predecir. Si es None, se usan las definidas en la configuración del modelo.
        patience (int, optional): Número de épocas sin mejora antes de detener el entrenamiento temprano.
        epochs (int, optional): Número máximo de épocas para entrenar.
        
    Returns:
        None
    """
    seed = 711  # Mantener una semilla fija para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Para GPU
        # Enable cuDNN benchmarking for improved performance with fixed input sizes
        torch.backends.cudnn.benchmark = True
        # Enable TF32 on Ampere GPUs for faster matrix operations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- Obtener Configuración Completa del Modelo ---
    config = get_model_config(model_name)
    inputs_from_config = config["features"]  # Columnas de entrada definidas para el modelo
    cols_from_config = config["cols"]  # Columnas a leer del CSV
    targets_from_config = config["targets"]  # Targets definidos para el modelo
    params = config["params"]  # Hiperparámetros del modelo
    stratify_cols = targets_from_config if len(targets_from_config) == 1 else config["stratify"]
    
    # --- Estratificación ---
    # -----------------------
    
    # Extraer hiperparámetros del diccionario 'params'
    epochs = EPOCHS
    patience = PATIENCE
    # --------------------------------------------------

    # Determinar las columnas target finales a usar
    target_cols = [t.lower() for t in targets_from_config]
    
    data_loader_obj = PGenDataProcess()  # 1. Crear el procesador de datos
    # 2. Cargar y limpiar el DataFrame COMPLETO
    try:
        df = data_loader_obj.load_data(
            csv_files,
            cols_from_config,
            inputs_from_config,
            targets_from_config,
            multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
            stratify_cols=stratify_cols,
        )
    except AttributeError as e:
        logger.error(f"Error: {e}")
        raise

    logger.info(f"Semilla utilizada: {seed}")
    
    # 3. Dividir ANTES de procesar (Usando estratificación)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2, # Fracción para validación
        random_state=seed,
        stratify=df["stratify_col"] # <-- Ahora usa 'effect_type'
    )
    val_df = val_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    logger.info(f"División estratificada completada. Train: {len(train_df)}, Val: {len(val_df)}")

    # 4. Ajustar (FIT) SÓLO con datos de entrenamiento
    data_loader_obj.fit(train_df)
    
    # 5. Transformar AMBOS sets por separado
    train_processed_df = data_loader_obj.transform(train_df)
    val_processed_df = data_loader_obj.transform(val_df)
    logger.info("Pre-procesamiento de datos completado.")
    
    logger.info("\nIniciando creación de DataLoaders...")
    # Crear Datasets y DataLoaders
    train_dataset = PGenDataset(
        train_processed_df,
        inputs_from_config, 
        target_cols, 
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df,
        inputs_from_config,
        target_cols, 
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    # Optimize attributes
    num_workers = min(multiprocessing.cpu_count(), 8)
    # Enable pin_memory for faster GPU transfer when CUDA is available
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
    )

    # --- Preparar Información para el Modelo ---
    n_targets_list = [len(data_loader_obj.encoders[tc].classes_) for tc in target_cols]
    target_dims = {
        col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)
    }

    # Obtener número de clases para los inputs
    n_features_list = [len(data_loader_obj.encoders[f].classes_) for f in inputs_from_config]
    feature_dims = {
        col_name: n_features_list[i] for i, col_name in enumerate(inputs_from_config)
    }

    # --- Instanciar el Modelo ---
    model = DeepFM_PGenModel(
        n_features=feature_dims,
        target_dims=target_dims,
        embedding_dim=params["embedding_dim"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        n_layers=params["n_layers"],
        #------------------------------------------
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.0),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 256),
        embedding_dropout=params.get("embedding_dropout", 0.1),
    )
    ######################## TESTEO ####################
    #PRETRAINED_PATH = Path(MODELS_DIR, "pmodel_Phenotype_Effect_Outcome.pth")
    #model.load_pretrained_embeddings(str(PRETRAINED_PATH), freeze=False)
    ######################## TESTEO ####################

    model = model.to(device)

    # --- Definir Optimizador y Criterios de Loss ---
    optimizer = create_optimizer(model, params)
    scheduler = create_scheduler(optimizer, params)

    criterions_list = create_criterions(target_cols, params, device)
    criterions = criterions_list + [optimizer]
    
    # ---------------------------------------------------------

    # --- Ejecutar Entrenamiento ---
    print(f"Iniciando entrenamiento final para {model_name} con Ponderación de Incertidumbre...")
    best_loss, best_accuracy_list, avg_per_task_losses = train_model( # type: ignore
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=epochs,
        patience=patience,
        model_name=model_name,
        feature_cols=inputs_from_config,
        device=device,
        target_cols=target_cols,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        params_to_txt=params,
        scheduler=scheduler,
        progress_bar=True,
        return_per_task_losses=True, 
    )

    # --- Guardar Resultados ---
    print(f"Entrenamiento completado. Mejor loss en validación: {best_loss:.5f}")
    
    save_model( model=model,
                target_cols=target_cols,
                best_loss=best_loss,
                best_accuracies=best_accuracy_list,
                avg_per_task_losses=avg_per_task_losses,
                model_name=model_name,
                params_to_txt=params
            )

    results_dir = Path(PGEN_MODEL_DIR, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = (
        results_dir / f"training_report_{model_name}_{round(best_loss, 4)}.txt"
    )

    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Mejor loss en validación: {best_loss:.5f}\n")
        if best_accuracy_list:
            avg_acc = sum(best_accuracy_list) / len(best_accuracy_list)
            f.write(f"Precisión Promedio (Avg Acc) en mejor época: {avg_acc:.4f}\n")
            f.write("Precisión por Tarea:\n")
            for i, col in enumerate(target_cols):
                f.write(f"  - {col}: {best_accuracy_list[i]:.4f}\n")
        if avg_per_task_losses:
            f.write("Pérdida Promedio por Tarea:\n")
            for i, col in enumerate(target_cols):
                f.write(f"  - {col}: {avg_per_task_losses[i]:.4f}\n")
        
        f.write("\nHiperparámetros Utilizados:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")


    print(f"Reporte de entrenamiento guardado en: {report_file}")

    # Guardar los encoders usados
    encoders_dir = Path(MODEL_ENCODERS_DIR)
    #models_dir = Path(MODELS_DIR)
    #model_picke_file = models_dir / f"pmodel_{model_name}_complete.pkl"
    if not encoders_dir.exists():
        encoders_dir.mkdir(parents=True, exist_ok=True)

    
    #print(f"Modelo_Completo guardado en: {model_picke_file}")

    encoders_file = encoders_dir / f"encoders_{model_name}.pkl"
    joblib.dump(data_loader_obj.encoders, encoders_file)
    print(f"Encoders guardados en: {encoders_file}")

    return model