#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Librerías estándar
import datetime
import json
import math
import random
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import List

# Librerías de terceros
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna import create_study
from optuna.storages import JournalStorage
from optuna.visualization import plot_optimization_history
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Librerías propias
from src.config.config import PGEN_MODEL_DIR, PROJECT_ROOT
from .data import PGenDataset, PGenDataProcess, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import MODEL_REGISTRY, MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model
# from src.scripts.custom_callback import TqdmCallback  # type: ignore

# --- Configuración ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna_results = Path(PROJECT_ROOT, "optuna_outputs", "figures")
optuna_results.mkdir(parents=True, exist_ok=True)

N_TRIALS = 100
EPOCH = 100
PATIENCE = 15


def get_optuna_params(trial: optuna.Trial, model_name: str) -> dict:
    """
    Sugiere hiperparámetros para un trial de Optuna basándose
    en la configuración dinámica de MODEL_REGISTRY.
    """
    model_conf = MODEL_REGISTRY.get(model_name)

    if not model_conf:
        raise ValueError(
            f"No se encontró la configuración para el modelo: {model_name}"
        )

    search_space = model_conf.get("params_optuna")
    if not search_space:
        print(
            f"Advertencia: No se encontró 'params_optuna' para '{model_name}'. "
            f"Usando 'params' fijos del registro."
        )
        return model_conf.get("params", {}).copy()

    suggested_params = {}

    for param_name, space_definition in search_space.items():
        if isinstance(space_definition, list):
            if space_definition and space_definition[0] == "int":
                try:
                    _, low, high = space_definition[:3]
                    step = space_definition[3] if len(space_definition) > 3 else 1
                    log = space_definition[4] if len(space_definition) > 4 else False
                    suggested_params[param_name] = trial.suggest_int(
                        param_name, int(low), int(high), step=int(step), log=bool(log)
                    )
                except (ValueError, TypeError, IndexError):
                    print(
                        f"Advertencia: Formato de lista 'int' incorrecto para '{param_name}'. "
                        f"Debe ser ['int', low, high, step(opcional)]. Omitiendo."
                    )
            else:
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, space_definition
                )
        elif isinstance(space_definition, tuple):
            low, high = space_definition
            is_log_scale = param_name in ["learning_rate", "weight_decay"]
            suggested_params[param_name] = trial.suggest_float(
                param_name, low, high, log=is_log_scale
            )
        else:
            print(
                f"Advertencia: Tipo de espacio de búsqueda desconocido para "
                f"'{param_name}': {type(space_definition)}. Omitiendo."
            )
            
    return suggested_params


def get_input_dims(data_loader: PGenDataProcess) -> dict:
    """
    Obtiene los tamaños de vocabulario (n_classes) para todas las
    columnas de entrada (inputs) requeridas por el modelo.
    """
    # Define las columnas de input que tu modelo SIEMPRE espera
    input_cols = ["drug", "gene", "allele", "variant/haplotypes"]
    
    dims = {}
    for col in input_cols:
        try:
            dims[col] = len(data_loader.encoders[col].classes_)
        except KeyError:
            raise KeyError(f"Error: No se encontró el encoder para la columna de input '{col}'.")
    
    return dims


def get_output_sizes(data_loader, target_cols):
    """Obtiene los tamaños de vocabulario para las columnas de salida (targets)."""
    return [len(data_loader.encoders[t].classes_) for t in target_cols]


def optuna_objective(
    trial: optuna.Trial, 
    model_name: str, 
    pbar: tqdm, 
    target_cols: List[str],
    
    # --- DATOS PRE-PROCESADOS ---
    fitted_data_loader: PGenDataProcess,
    train_processed_df: pd.DataFrame,
    val_processed_df: pd.DataFrame,
    csvfiles: Path | str, # Solo para el print
    class_weights_task3: torch.Tensor = None #type: ignore
):
    """
    Función objetivo de Optuna.
    Recibe datos pre-procesados y se encarga del modelado y entrenamiento.
    """
    
    seed = 711
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    
    
    # --- 1. Definir Hiperparámetros ---
    params = get_optuna_params(trial, model_name)
    print(params)
    
    # --- 2. Cargar Datos ---
    # Los datos ya están listos. Solo creamos Datasets y DataLoaders.
    train_dataset = PGenDataset(
        train_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_processed_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # --- 3. Definir Modelo (Dinámicamente) ---
    input_dims = get_input_dims(fitted_data_loader)
    n_targets_list = get_output_sizes(fitted_data_loader, target_cols)
    target_dims = {
        col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)
    }

    if trial.number == 0:
        print(f" Cargando datos de: {Path(csvfiles)}")
    
    model = DeepFM_PGenModel(
        n_drugs=input_dims["drug"],
        n_genotypes=input_dims["variant/haplotypes"],
        n_genes=input_dims["gene"],
        n_alleles=input_dims["allele"],
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"], # AÑADIDO TESTEO
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        target_dims=target_dims,
    )
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # --- 4. Definir Criterio (Dinámicamente) ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-5), # Usar 'get' para un default
    )

    criterions = []
    '''
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=0.1))
    '''
    
    task3_name = 'effect_type' # El mismo nombre
    
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        
        # Aplicar los pesos SUAVES si existen
        elif col == task3_name and class_weights_task3 is not None:
            criterions.append(nn.CrossEntropyLoss(
                label_smoothing=0.1, 
                weight=class_weights_task3 # <-- Aplicar pesos
            ))
            if trial.number == 0:
                print(f"Optuna: Aplicando WeightedCrossEntropyLoss SUAVE para '{task3_name}'.")
        
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=0.1))
    
    criterions.append(optimizer)

    
    # --- 5. Entrenar ---
    best_loss, best_accuracies_list = train_model(
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=EPOCH,
        patience=PATIENCE,
        target_cols=target_cols,
        scheduler=None,
        params_to_txt=params,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        progress_bar=False,
        model_name=model_name,
        use_weighted_loss=False # Usamos suma simple para Optuna
    )

    # --- 6. Calcular Métricas y Guardar en Trial ---
    max_loss_list = []
    for i, col in enumerate(target_cols):
        if col in MULTI_LABEL_COLUMN_NAMES:
            max_loss_list.append(math.log(2))
        else:
            n_classes = n_targets_list[i]
            max_loss_list.append(math.log(n_classes) if n_classes > 1 else 0.0)

    max_loss = sum(max_loss_list) if max_loss_list else 1.0
    normalized_loss = best_loss / max_loss if max_loss > 0 else best_loss
    avg_accuracy = (
        sum(best_accuracies_list) / len(best_accuracies_list)
        if best_accuracies_list
        else 0.0
    )

    trial.set_user_attr("avg_accuracy", avg_accuracy)
    trial.set_user_attr("normalized_loss", normalized_loss)
    trial.set_user_attr("all_accuracies", best_accuracies_list)
    trial.set_user_attr("seed", seed)

    info_dict = {
        "Best_Loss": f"{best_loss:.4f}",
        "Trial": trial.number,
        "ValLoss": f"{best_loss:.4f}",
        "NormLoss": f"{normalized_loss:.4f}",
        "AvgAcc": f"{avg_accuracy:.4f}",
        "BS": params["batch_size"],
        "LR": f"{params['learning_rate']:.6f}",
    }
    pbar.set_postfix(info_dict, "\b") # type: ignore
    pbar.update(1)

    return best_loss


def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    """Obtiene los 'n' mejores trials válidos de un estudio."""
    valid_trials = [
        t
        for t in study.trials
        if t.value is not None and isinstance(t.value, (int, float))
    ]
    # Manejar estudios multi-objetivo (aunque aquí no se usa)
    key_func = lambda t: float(t.values[0] if isinstance(t.values, list) else t.value)
    return sorted(valid_trials, key=key_func)[:n]


def _save_optuna_report(
    study: optuna.Study,
    model_name: str,
    results_file_json: Path,
    results_file_txt: Path,
    top_5_trials: List[optuna.trial.FrozenTrial],
):
    """Función auxiliar para guardar los reportes JSON y TXT."""

    if not top_5_trials:
        print(
            "Advertencia: No se encontraron trials válidos para guardar en el reporte.",
            file=sys.stderr,
        )
        return

    best_trial = top_5_trials[0]
    normalized_loss = best_trial.user_attrs.get("normalized_loss", 0.0)
    avg_accuracy = best_trial.user_attrs.get("avg_accuracy", 0.0)

    # --- Guardar JSON ---
    output = {
        "model": model_name,
        "best_trial": {
            "loss": best_trial.value,
            "normalized_loss": normalized_loss,
            "avg_accuracy": avg_accuracy,
            "params": best_trial.params,
            "all_accuracies": best_trial.user_attrs.get("all_accuracies", []),
        },
        "top5": [{"loss": t.value, "params": t.params} for t in top_5_trials],
        "n_trials": len(study.trials),
    }
    with open(results_file_json, "w") as f:
        json.dump(output, f, indent=2)

    # --- Guardar TXT ---
    with open(results_file_txt, "w") as f2:
        f2.write(
            f"Mejores hiperparámetros para {model_name} (loss: {best_trial.value}, normalized_loss: {normalized_loss:.4f}, avg_accuracy: {avg_accuracy:.4f}):\n"
        )
        for key, value in best_trial.params.items():
            f2.write(f"  {key}: {value}\n")
        f2.write("\nTop 5 pruebas:\n")
        for i, t in enumerate(top_5_trials, 1):
            f2.write(f"{i}. Loss: {t.value}, Params: {t.params}\n")

    print(f"\nReportes de Optuna guardados en: {results_file_json.parent}")


def run_optuna_with_progress(
    model_name, n_trials=N_TRIALS, output_dir=PGEN_MODEL_DIR, target_cols=None
):
    """
    Ejecuta un estudio de Optuna, cargando los datos
    y guardando los reportes.
    """
    datetime_study = datetime.datetime.now().strftime("%d_%m_%y__%H_%M")

    output_dir = Path(output_dir)
    results_dir = output_dir / "optuna_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    seed = 711 # Usar la misma semilla

    # --- 1. Definir Targets ---
    config = get_model_config(model_name)
    targets_from_config = [t.lower() for t in config["targets"]]
    cols_from_config = config["cols"]

    if target_cols is None:
        target_cols = targets_from_config

    # --- 2. Cargar y Pre-procesar Datos (UNA SOLA VEZ) ---
    print("Cargando y pre-procesando datos (una sola vez)...")
    csvfiles, _ = train_data_import(targets_from_config)
    
    data_loader = PGenDataProcess()
    
    # Usamos load_data (que asumo que es tu método refactorizado)
    df = data_loader.load_data(
        csvfiles,
        cols_from_config,
        targets_from_config,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=['effect_type'],
    )
    
    # Limpiar NaNs triviales
    df = df.dropna(subset=['phenotype_outcome', 'effect_type'])

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["stratify_col"]
    )
    
    data_loader.fit(train_df)
    train_processed_df = data_loader.transform(train_df)
    val_processed_df = data_loader.transform(val_df)
    print("Datos listos para la optimización.")
    
    '''TESTEO DE NUEVO'''
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_task3 = None
    task3_name = 'effect_type'

    if task3_name in target_cols:
        try:
            encoder_task3 = data_loader.encoders[task3_name]
            counts = train_processed_df[task3_name].value_counts().sort_index()
            all_counts = torch.zeros(len(encoder_task3.classes_))
            for class_id, count in counts.items():
                all_counts[int(class_id)] = count
            
            # --- ¡LA FÓRMULA SUAVE! ---
            # Opción A: Log Smoothing (Recomendada)
            weights = 1.0 / torch.log(all_counts + 2) # +2 para evitar log(1)=0
            
            # Opción B: Sqrt Smoothing (Más fuerte pero segura)
            # weights = 1.0 / torch.sqrt(all_counts + 1)
            
            # Normalizar para que la media sea 1
            weights = weights / weights.mean() 
            class_weights_task3 = weights.to(device)
            print(f"Pesos de clase SUAVES para '{task3_name}' calculados.")

        except Exception as e:
            print(f"Error calculando pesos suaves de clase: {e}")
    
    '''FIN DE TESTEO'''
    

    # --- 3. Ejecutar Estudio ---
    pbar = tqdm(total=n_trials, file=sys.stderr, desc=f"Optuna ({model_name})", colour="green", nrows=3)

    # Pasamos los datos pre-procesados a la función objetivo
    optuna_func = partial(
        optuna_objective, 
        model_name=model_name, 
        target_cols=target_cols, 
        pbar=pbar,
        fitted_data_loader=data_loader,
        train_processed_df=train_processed_df,
        val_processed_df=val_processed_df,
        csvfiles=csvfiles, # type: ignore
        class_weights_task3=class_weights_task3 # type: ignore
    )

    study = optuna.create_study(
        direction="minimize", study_name=f"optuna_{model_name}_{datetime_study}"
    )

    study.optimize(
        optuna_func,
        n_trials=n_trials,
    )
    pbar.close()
    
    # --- 4. Guardar Gráfico de Historial (Plotly) ---
    '''
    if study.trials:
        fig = plot_optimization_history(study)
        graph_filename = Path(optuna_results, f"history_{model_name}_{datetime_study}.html")
        fig.write_html(str(graph_filename))
        fig.write_image(str(graph_filename.with_suffix(".png")))
        print(f"Gráfico de historial de optimización guardado en: {graph_filename}")
    '''
    
    # --- 5. Obtener y Guardar Reporte ---
    best_5 = get_best_trials(study, 5)

    if not best_5:
        print(
            "Optimización de Optuna completada, pero no se encontraron trials exitosos.",
            file=sys.stderr,
        )
        return {}, None, None, None # Devuelve tupla vacía

    filename = f"optuna_{model_name}_{datetime_study}"
    results_file_json = Path(results_dir / f"{filename}.json")
    results_file_txt = Path(results_dir / f"{filename}.txt")
    
    _save_optuna_report(study, model_name, results_file_json, results_file_txt, best_5)

    # Extraer métricas del mejor trial para el return
    best_trial = best_5[0]
    best_loss = best_trial.value
    best_params = best_trial.params
    normalized_loss = best_trial.user_attrs.get("normalized_loss", 0.0)

    return best_params, best_loss, results_file_json, normalized_loss