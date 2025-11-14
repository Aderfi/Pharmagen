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

from .data import PGenDataset, PGenDataProcess, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import CLINICAL_PRIORITIES, MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model, save_model
from .train_utils import create_optimizer, create_scheduler, create_criterions
from .focal_loss import FocalLoss

logger = logging.getLogger(__name__)

# Constantes de configuración por defecto
EPOCHS = 100
PATIENCE = 25

# <--- CORRECCIÓN 1: Firma de la función ---
# La firma ahora acepta 'params' (un dict) como lo pasa __main__.py.
# Los argumentos 'epochs', 'patience', etc., se eliminan de aquí
# porque ya están DENTRO de 'params'.
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

    # --- Obtener Configuración Completa del Modelo ---
    config = get_model_config(model_name)
    inputs_from_config = config["inputs"]  # Columnas de entrada definidas para el modelo
    cols_from_config = config["cols"]  # Columnas a leer del CSV
    targets_from_config = config["targets"]  # Targets definidos para el modelo
    
    # <--- CORRECCIÓN 2: Estratificación ---
    # 'phenotype_outcome' es multi-label y fallará.
    # Usamos 'effect_type' que es single-label y tiene pocos NaNs (que limpiamos después).
    #stratify_cols = ['phenotype_outcome']
    stratify_cols = ['effect_type']
    
    params = config["params"]  # <--- ELIMINADO: 'params' ahora se pasa como argumento.
    # -----------------------------------------------
    
    # Extraer hiperparámetros del diccionario 'params'
    epochs = EPOCHS
    patience = PATIENCE
    # --------------------------------------------------

    # Determinar las columnas target finales a usar
    if target_cols is None:
        target_cols = [t.lower() for t in targets_from_config]
    else:
        target_cols = [t.lower() for t in target_cols]

    # --- Cargar y Preparar Datos ---
    actual_csv_files = csv_files

    data_loader_obj = PGenDataProcess()  # 1. Crear el procesador

    # 2. Cargar y limpiar el DataFrame COMPLETO
    try:
        df = data_loader_obj.load_data(
            actual_csv_files,
            cols_from_config,
            inputs_from_config,
            targets_from_config,
            multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
            stratify_cols=stratify_cols,
        )
    except AttributeError as e:
        logger.error(f"Error: {e}")
        logger.error("Asegúrate de que PGenDataProcess tiene el método 'load_data' (o 'load_and_clean_data')")
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
    
    """    aqui camios"""
    
    class_weights_task3 = None
    task3_name = 'effect_type'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if task3_name in target_cols:
        print(f"Calculando pesos de clase SUAVES para '{task3_name}'...")
        try:
            # Debug: Verificar qué encoders están disponibles
            print(f"DEBUG: Encoders disponibles: {list(data_loader_obj.encoders.keys())}")
            print(f"DEBUG: target_cols = {target_cols}")
            
            encoder_task3 = data_loader_obj.encoders[task3_name]
            print(f"DEBUG: Número de clases en encoder '{task3_name}': {len(encoder_task3.classes_)}")
            print(f"DEBUG: Clases: {encoder_task3.classes_}")
            
            counts = train_processed_df[task3_name].value_counts().sort_index()
            print(f"DEBUG: Distribución de clases en datos: {counts}")
            
            all_counts = torch.zeros(len(encoder_task3.classes_))
            for class_id, count in counts.items():
                all_counts[int(class_id)] = count # type: ignore
            
            # Usar la fórmula de Log Smoothing que funcionó
            weights = 1.0 / torch.log(all_counts + 2) # +2 para evitar log(1)=0
            weights = weights / weights.mean() # Normalizar
            class_weights_task3 = weights.to(device)
            print(f"DEBUG: Shape de class_weights_task3: {class_weights_task3.shape}")
            print("Pesos de clase listos para el entrenamiento final.")
        except Exception as e:
            print(f"Error calculando pesos de clase en pipeline: {e}")
            import traceback
            traceback.print_exc()
    
    """fin cambios"""
    
    # --- FIN DE LA CORRECCIÓN DE DATOS ---

    # Crear Datasets y DataLoaders
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    # Optimize num_workers based on available CPU cores
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
    n_drugs = len(data_loader_obj.encoders["drug"].classes_)
    n_genalle = len(data_loader_obj.encoders["genalle"].classes_) # cambiado a Genalle
    n_genes = len(data_loader_obj.encoders["gene"].classes_)
    n_alleles = len(data_loader_obj.encoders["allele"].classes_)

    # --- Instanciar el Modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM_PGenModel(
        n_drugs,
        n_genalle,
        n_genes, 
        n_alleles,
        params["embedding_dim"],
        params["n_layers"],
        params["hidden_dim"],
        params["dropout_rate"],
        target_dims=target_dims,  # Dinámico #type: ignore
        ############
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

    criterions_list = create_criterions(target_cols, params, class_weights_task3, device)
    criterions = criterions_list + [optimizer]
    
    # ---------------------------------------------------------

    # --- Ejecutar Entrenamiento ---
    print(f"Iniciando entrenamiento final para {model_name} con Ponderación de Incertidumbre...")
    best_loss, best_accuracy_list, avg_per_task_losses = train_model( #type: ignore
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=epochs,
        patience=patience,
        model_name=model_name,
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

    results_dir = Path(PGEN_MODEL_DIR, "results")  # Usar Path para consistencia
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