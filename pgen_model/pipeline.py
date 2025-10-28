# coding=utf-8
import os
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from src.config.config import MODEL_ENCODERS_DIR
from torch.utils.data import DataLoader

from .data import PGenDataset, PGenInputDataset, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model


def train_pipeline(
    PGEN_MODEL_DIR,
    csv_files,
    model_name,
    params,
    epochs=100,
    patience=20,
    batch_size=8,
    target_cols=None,
):
    """
    Función principal para entrenar un modelo PGen.

    Args:
        PMODEL_DIR (str): Directorio base para guardar modelos y resultados.
        csv_files (str/Path): Ruta al archivo CSV de datos. (Actualmente hardcodeado)
        model_name (str): Nombre del modelo a entrenar (clave en MODEL_REGISTRY).
        params (dict): Diccionario con los hiperparámetros del modelo.
        epochs (int): Número máximo de épocas.
        patience (int): Paciencia para early stopping.
        batch_size (int): Tamaño del lote.
        target_cols (list, optional): Lista de columnas target a usar. Si es None,
                                      se usan las definidas en model_configs.py.

    Returns:
        torch.nn.Module: El modelo entrenado.
    """
    seed = 711  # Mantener una semilla fija para reproducibilidad
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # Para GPU

    # --- Obtener Configuración Completa del Modelo ---
    config = get_model_config(model_name)
    cols_from_config = config["cols"]  # Columnas a leer del CSV
    targets_from_config = config["targets"]  # Targets definidos para el modelo
    # --------------------------------------------------

    # Determinar las columnas target finales a usar
    if target_cols is None:
        target_cols = [t.lower() for t in targets_from_config]
    else:
        target_cols = [t.lower() for t in target_cols]

    # --- Cargar y Preparar Datos ---
    # train_data_import ahora devuelve csvfiles, read_cols, equivalencias
    # 'read_cols' se usa internamente en train_data_import si es necesario
    actual_csv_files, cols_to_read = train_data_import(
        targets_from_config
    )
    # Nota: 'csv_files' pasado como argumento no se usa si train_data_import lo redefine

    data_loader_obj = PGenInputDataset()  # Renombrado para claridad

    # Pasar equivalencias a load_data
    df = data_loader_obj.load_data(
        actual_csv_files,  # Usar los archivos definidos en train_data_import
        cols_from_config,  # Usar las columnas de la config
        targets_from_config,  # Usar los targets de la config
        #equivalencias,  # Pasar equivalencias
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
    )

    print(f"Semilla utilizada: {seed}")
    # Separar train/validation (estratificado si es posible, aunque aquí es aleatorio)
    # Considerar StratifiedShuffleSplit si el dataset es grande y desbalanceado
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index).reset_index(
        drop=True
    )  # Reset index es buena práctica
    train_df = train_df.reset_index(drop=True)

    # Crear Datasets y DataLoaders
    train_dataset = PGenDataset(
        train_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )  # Optimizaciones
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )  # No shuffle en validación

    # --- Preparar Información para el Modelo ---
    # Obtener número de clases/dimensiones para los outputs
    # Usar data_loader_obj que tiene los encoders ajustados
    n_targets_list = [len(data_loader_obj.encoders[tc].classes_) for tc in target_cols]
    target_dims = {
        col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)
    }

    # Obtener número de clases para los inputs
    n_drugs = len(data_loader_obj.encoders["drug"].classes_)
    n_genes = len(data_loader_obj.encoders["gene"].classes_)
    n_alleles = len(data_loader_obj.encoders["allele"].classes_)
    n_genotypes = len(data_loader_obj.encoders["genotype"].classes_)

    # --- Instanciar el Modelo ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFM_PGenModel(
        n_drugs,
        n_genes,
        n_genotypes,  # n_alleles,
        params["embedding_dim"],
        params["hidden_dim"],
        params["dropout_rate"],
        target_dims=target_dims,  # Dinámico #type: ignore
    )
    model = model.to(device)

    # --- Definir Optimizador y Criterios de Loss ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-5),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5)

    criterions_list = []
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions_list.append(nn.BCEWithLogitsLoss())
        else:
            criterions_list.append(nn.CrossEntropyLoss(label_smoothing=0.1))

    criterions = criterions_list + [optimizer]  # Combinar criterios y optimizador
    # ---------------------------------------------------------

    # --- Ejecutar Entrenamiento ---
    print(f"Iniciando entrenamiento para {model_name}...")
    best_loss, best_accuracy_list = train_model(
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=epochs,
        patience=patience,
        target_cols=target_cols,
        device=device,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        #weights_dict=weights, # type: ignore
        params_to_txt=params,
        # scheduler se puede añadir aquí si se desea para el run final
    )

    # --- Guardar Resultados ---""""
    print(f"Entrenamiento completado. Mejor loss en validación: {best_loss:.5f}")

    results_dir = Path(PGEN_MODEL_DIR, "results")  # Usar Path para consistencia
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = (
        results_dir / f"training_report_{model_name}_{round(best_loss, 4)}.txt"
    )  # Nombre de archivo específico

    with open(report_file, "w") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"Mejor loss en validación: {best_loss:.5f}\n")
        # Calcular y escribir Avg Accuracy si best_accuracy_list no está vacío
        if best_accuracy_list:
             avg_acc = sum(best_accuracy_list) / len(best_accuracy_list)
             f.write(f"Precisión Promedio (Avg Acc) en mejor época: {avg_acc:.4f}\n")
             f.write("Precisión por Tarea:\n")
             for i, col in enumerate(target_cols):
                 f.write(f"  - {col}: {best_accuracy_list[i]:.4f}\n")
        f.write("\nHiperparámetros Utilizados:\n")
        for key, value in params.items():
            f.write(f"  {key}: {value}\n")
        """
        # Guardar también los pesos usados
        f.write("\nPesos de Loss Utilizados:\n")
        if weights:
            for key, value in weights.items():
                 # Solo mostrar pesos de los targets actuales
                 if key.lower() in target_cols:
                     f.write(f"  {key}: {value}\n")
        """
       
            
        
        f.write("\nEstrategia de Ponderación:\n")
        # Asumiendo que ahora SIEMPRE usas UW
        f.write("  Uncertainty Weighting (Pesos dinámicos aprendibles)\n")

        # 💡 Opcional: Escribir los pesos finales aprendidos por UW
    
        f.write("  Pesos Efectivos Finales (exp(-log_sigma^2)):\n")
        for name, log_sigma in model.log_sigmas.items():
            final_weight = torch.exp(-log_sigma).item()
            f.write(f"  - {name}: {final_weight:.4f}\n")
    
    print(f"Reporte de entrenamiento guardado en: {report_file}")

    # Guardar los encoders usados

    encoders_dir = Path(MODEL_ENCODERS_DIR)
    if not encoders_dir.exists():
        encoders_dir.mkdir(parents=True, exist_ok=True)

    encoders_file = encoders_dir / f"encoders_{model_name}.pkl"
    # joblib.dump(data_loader_obj.encoders, encoders_file)
    # print(f"Encoders guardados en: {encoders_file}")

    return model
    # joblib.dump(data_loader_obj.encoders, results_dir / f'encoders_{model_name}.pkl')
