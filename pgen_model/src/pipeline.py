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
from sklearn.model_selection import train_test_split 

from .data import PGenDataset, PGenDataProcess, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import CLINICAL_PRIORITIES, MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model, save_model
from .focal_loss import FocalLoss

EPOCHS = 150
PATIENCE = 30

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

    # --- Obtener Configuración Completa del Modelo ---
    config = get_model_config(model_name)
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
            targets_from_config,
            multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
            stratify_cols=stratify_cols,
        )
    except AttributeError as e:
        print(f"Error: {e}")
        print("Asegúrate de que PGenDataProcess tiene el método 'load_data' (o 'load_and_clean_data')")
        raise

    print(f"Semilla utilizada: {seed}")
    
    # 3. Dividir ANTES de procesar (Usando estratificación)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2, # Fracción para validación
        random_state=seed,
        stratify=df["stratify_col"] # <-- Ahora usa 'effect_type'
    )
    val_df = val_df.reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    print(f"División estratificada completada. Train: {len(train_df)}, Val: {len(val_df)}")

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
            encoder_task3 = data_loader_obj.encoders[task3_name]
            counts = train_processed_df[task3_name].value_counts().sort_index()
            all_counts = torch.zeros(len(encoder_task3.classes_))
            for class_id, count in counts.items():
                all_counts[int(class_id)] = count
            
            # Usar la fórmula de Log Smoothing que funcionó
            weights = 1.0 / torch.log(all_counts + 2) # +2 para evitar log(1)=0
            weights = weights / weights.mean() # Normalizar
            class_weights_task3 = weights.to(device)
            print("Pesos de clase listos para el entrenamiento final.")
        except Exception as e:
            print(f"Error calculando pesos de clase en pipeline: {e}")
    
    """fin cambios"""
    
    # --- FIN DE LA CORRECCIÓN DE DATOS ---

    # Crear Datasets y DataLoaders
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=4,
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
    task3_name = 'effect_type'
    
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions_list.append(nn.BCEWithLogitsLoss())
        
        elif col == task3_name and class_weights_task3 is not None:
            # ✅ FOCAL LOSS CON GAMMA AJUSTADO
            # Dado que la loss es MUY alta (2.81), usar gamma alto
            criterions_list.append(FocalLoss(
                alpha=class_weights_task3,
                gamma=2.5,  # ✅ Más agresivo que el estándar (2.0)
                label_smoothing=0.15,  # ✅ Más smoothing para regularizar
            ))
            print(f"🔥 Aplicando Focal Loss (γ=2.5) + Class Weighting a '{task3_name}'")
            print(f"   Baseline Loss: 2.8147 → Target: <2.0")

    else:
        criterions_list.append(nn.CrossEntropyLoss(label_smoothing=0.1))

    criterions = criterions_list + [optimizer]
    
    """
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions_list.append(nn.BCEWithLogitsLoss())
        
        # Aplicar los pesos de clase a la Tarea 3
        elif col == task3_name and class_weights_task3 is not None:
            criterions_list.append(nn.CrossEntropyLoss(
                label_smoothing=0.1,
                weight=class_weights_task3
            ))
            print(f"Aplicando WeightedCrossEntropyLoss (Class Weighting) a '{task3_name}'.")

        else: # Tareas 1 y 2 (single-label sin pesos de clase)
            criterions_list.append(nn.CrossEntropyLoss(label_smoothing=0.1))

    criterions = criterions_list + [optimizer]
    
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions_list.append(nn.BCEWithLogitsLoss())
        else:
            criterions_list.append(nn.CrossEntropyLoss(label_smoothing=0.1))

    criterions = criterions_list + [optimizer]  # Combinar criterios y optimizador
    """
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
        use_weighted_loss=True,
        task_priorities=CLINICAL_PRIORITIES,
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

        f.write("\nEstrategia de Ponderación:\n")
        f.write("  Uncertainty Weighting (Pesos dinámicos aprendibles)\n")

        if hasattr(model, 'log_sigmas'):
            f.write("  Pesos Efectivos Finales (exp(-log_sigma^2)):\n")
            for name, log_sigma in model.log_sigmas.items():
                if name in target_dims:
                    final_weight = torch.exp(-log_sigma).item()
                    f.write(f"  - {name}: {final_weight:.4f}\n")
        else:
            f.write("  Ponderación de Incertidumbre no activada (log_sigmas no encontrado).\n")

    print(f"Reporte de entrenamiento guardado en: {report_file}")

    # Guardar los encoders usados
    encoders_dir = Path(MODEL_ENCODERS_DIR)
    if not encoders_dir.exists():
        encoders_dir.mkdir(parents=True, exist_ok=True)

    encoders_file = encoders_dir / f"encoders_{model_name}.pkl"
    joblib.dump(data_loader_obj.encoders, encoders_file)
    print(f"Encoders guardados en: {encoders_file}")

    return model