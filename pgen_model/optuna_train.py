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

import numpy as np
import optuna
import pandas as pd

# Librerias de terceros
import torch
import torch.nn as nn
from optuna import create_study
from optuna.storages import JournalStorage
from src.config.config import PGEN_MODEL_DIR
from src.scripts.custom_callback import TqdmCallback  # type: ignore
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import PGenDataset, PGenInputDataset, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import MODEL_REGISTRY, MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model

# --- CORRECCIÓN 2: 'gelu' global eliminado por ser innecesario ---
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 300
EPOCH = 120
PATIENCE = 25


def get_optuna_params(trial: optuna.Trial, model_name: str) -> dict:
    """
    Sugiere hiperparámetros para un trial de Optuna basándose
    en la configuración dinámica de MODEL_REGISTRY.

    Maneja 3 tipos de definiciones en 'params_optuna':
    - list (normal):   Usa suggest_categorical (ej. [32, 64, 128])
    - list (híbrida):  Usa suggest_int si la lista es ["int", low, high, step(opc)]
    - tuple:           Usa suggest_float (ej. (1e-5, 1e-3))
    """
    model_conf = MODEL_REGISTRY.get(model_name)

    if not model_conf:
        raise ValueError(
            f"No se encontró la configuración para el modelo: {model_name}"
        )

    if "params_optuna" not in model_conf:
        print(
            f"Advertencia: No se encontró 'params_optuna' para '{model_name}'. "
            f"Usando 'params' fijos del registro."
        )
        return model_conf.get("params", {}).copy()

    search_space = model_conf["params_optuna"]
    suggested_params = {}

    # Iterar dinámicamente sobre los parámetros definidos
    for param_name, space_definition in search_space.items():

        # --- Caso A: El espacio es una LISTA ---
        if isinstance(space_definition, list):
            
            # --- (NUEVO) Sub-Caso A.1: Lista híbrida para suggest_int ---
            # Comprobamos si la lista sigue el formato ["int", low, high, step(opcional)]
            if space_definition and space_definition[0] == "int":
                try:
                    # Extraer low, high, y step opcional
                    _, low, high = space_definition[:3]
                    step = space_definition[3] if len(space_definition) > 3 else 1
                    log = space_definition[4] if len(space_definition) > 4 else False

                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        int(low),
                        int(high),
                        step=int(step),
                        log=bool(log)
                    )
                except (ValueError, TypeError, IndexError):
                    print(
                        f"Advertencia: Formato de lista 'int' incorrecto para '{param_name}'. "
                        f"Debe ser ['int', low, high, step(opcional)]. Omitiendo."
                    )

            # --- Sub-Caso A.2: Lista normal para suggest_categorical ---
            else:
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name,
                    space_definition,
                )

        # --- Caso B: El espacio es una TUPLA -> Usar suggest_float ---
        elif isinstance(space_definition, tuple):
            low, high = space_definition
            is_log_scale = param_name in ["learning_rate", "weight_decay"]
            suggested_params[param_name] = trial.suggest_float(
                param_name,
                low,
                high,
                log=is_log_scale,
            )

        else:
            print(
                f"Advertencia: Tipo de espacio de búsqueda desconocido para "
                f"'{param_name}': {type(space_definition)}. Omitiendo."
            )
            
    return suggested_params

def get_num_drugs(data_loader):
    return len(data_loader.encoders["drug"].classes_)


def get_num_genes(data_loader):
    return len(data_loader.encoders["gene"].classes_)


def get_num_alleles(data_loader):
    return len(data_loader.encoders["allele"].classes_)


def get_num_genotypes(data_loader):
    return len(data_loader.encoders["genotype"].classes_)


def get_output_sizes(data_loader, target_cols):
    # Esto sigue funcionando, ya que MultiLabelBinarizer.classes_
    # devuelve el número de clases (el ancho del vector binario).
    return [len(data_loader.encoders[t].classes_) for t in target_cols]


def optuna_objective(trial, model_name, pbar, target_cols=None):

    seed = 711
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Obtención de targets/columnas y pesos desde la configuración del modelo
    config = get_model_config(model_name)
    targets_from_config = [t.lower() for t in config["targets"]]
    weights_dict = config["weights"]

    if target_cols is None:
        target_cols = targets_from_config

    # --- 1. Definir Hiperparámetros ---
    # (Dejo tu bloque de 'intento modelo 2' activo)

    params = get_optuna_params(trial, model_name)
    print(params)
    params_to_txt = params  # Guardar para reporte
    numero_de_iter = trial.number  # type: ignore
    
    
    # --- 2. Cargar Datos ---
    csvfiles, cols_to_read= train_data_import(targets_from_config)
    data_loader = PGenInputDataset()

    df = data_loader.load_data(
        csvfiles,
        cols_to_read,
        targets_from_config,
        #equivalencias,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
    )

    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_df = df.sample(frac=0.8, random_state=seed)
    val_df = df.drop(train_df.index)

    train_dataset = PGenDataset(
        train_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )
    val_dataset = PGenDataset(
        val_df, target_cols, multi_label_cols=MULTI_LABEL_COLUMN_NAMES
    )

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=params["batch_size"])

    # --- 3. Definir Modelo (Dinámicamente) ---
    n_drugs = get_num_drugs(data_loader)
    n_genes = get_num_genes(data_loader)
    # n_alleles = get_num_alleles(data_loader)
    n_genotypes = get_num_genotypes(data_loader)

    n_targets_list = get_output_sizes(data_loader, target_cols)
    num_targets = len(target_cols)

    target_dims = {}
    for i, col_name in enumerate(target_cols):
        target_dims[col_name] = n_targets_list[i]

    
    if trial.number == 0:
            if isinstance(csvfiles, list):
                for i in csvfiles:
                    print(f" Cargando datos de: {Path(i)}")
            elif csvfiles:
                print(f" Cargando datos de: {Path(csvfiles)}")
    

    model = DeepFM_PGenModel(
        n_drugs,
        n_genes,
        n_genotypes,  # n_alleles,
        params["embedding_dim"],
        params["hidden_dim"],
        params["dropout_rate"],
        target_dims=target_dims,
    )

    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # --- 4. Definir Criterio (Dinámicamente) ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    criterions = []
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=0.1))

    criterions.append(optimizer)  # Añadir el optimizador al final

    epochs = EPOCH
    patience = PATIENCE

    
    
    # --- 5. Entrenar ---
    
    best_loss, best_accuracies_list = train_model(
        train_loader,
        val_loader,
        model,
        criterions,
        epochs=epochs,
        patience=patience,
        target_cols=target_cols,
        scheduler=None,
        params_to_txt=params_to_txt,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        #weights_dict=weights_dict,
        progress_bar=False,
        model_name=model_name
    )

    # --- 6. Calcular Métricas y Guardar en Trial ---

    # --- CORRECCIÓN 4: Cálculo dinámico de max_loss ---
    max_loss_list = []
    for i, col in enumerate(target_cols):
        if col in MULTI_LABEL_COLUMN_NAMES:
            # Para BCE, el loss de una predicción aleatoria (p=0.5) es log(2)
            max_loss_list.append(math.log(2))
        else:
            # Para CE, el loss de una predicción aleatoria es log(n_classes)
            n_classes = n_targets_list[i]
            # Evitar log(1) o log(0) si algo está mal
            if n_classes > 1:
                max_loss_list.append(math.log(n_classes))
            else:
                max_loss_list.append(0.0)  # Un target con 1 clase no tiene loss

    if not max_loss_list:  # Evitar división por cero si no hay targets
        max_loss = 1.0
    else:
        max_loss = sum(max_loss_list) / len(max_loss_list)
    # --- FIN DE CORRECCIÓN ---

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
    pbar.set_postfix(info_dict, "\b")
    pbar.update(1)

    return best_loss


def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    """Obtiene los 'n' mejores trials válidos de un estudio."""
    valid_trials = [
        t
        for t in study.trials
        if t.value is not None and isinstance(t.value, (int, float))
    ]
    return sorted(valid_trials, key=lambda t: float(t.values[0] if isinstance(t.values, list) else t.value))[:n]  # type: ignore


def _save_optuna_report(
    study: optuna.Study,
    model_name: str,
    results_path: Path,
    results_file_name: Path,
    top_5_trials: List[optuna.trial.FrozenTrial],
):
    """Función auxiliar para guardar los reportes JSON y TXT."""

    if not top_5_trials:
        print(
            "Advertencia: No se encontraron trials válidos para guardar en el reporte.",
            file=sys.stderr,
        )
        return

    best_trial = top_5_trials[0]  # El mejor es el primero de la lista ordenada

    # Extraer métricas guardadas del mejor trial
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
    with open(results_file_name.with_suffix(".json"), "w") as f:
        json.dump(output, f, indent=2)

    # --- Guardar TXT ---
    with open(results_file_name.with_suffix(".txt"), "w") as f:
        f.write(
            f"Mejores hiperparámetros para {model_name} (loss: {best_trial.value}, normalized_loss: {normalized_loss:.4f}, avg_accuracy: {avg_accuracy:.4f}):\n"
        )
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nTop 5 pruebas:\n")
        for i, t in enumerate(top_5_trials, 1):
            f.write(f"{i}. Loss: {t.value}, Params: {t.params}\n")

    print(f"\nReportes de Optuna guardados en: {results_file_name.parent}")


def run_optuna_with_progress(
    model_name, n_trials=N_TRIALS, output_dir=PGEN_MODEL_DIR, target_cols=None
):
    """
    Ejecuta un estudio de Optuna y guarda los reportes.
    """
    datestudy = datetime.date.today()
    timestudy = datetime.datetime.now().strftime("%H:%M")

    output_dir = Path(output_dir)
    results_dir = output_dir / "optuna_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file_json = Path(results_dir / f"optuna_{model_name}.json")
    results_file_txt = Path(results_dir / f"optuna_{model_name}.txt")

    # --- 1. Definir Targets ---
    config = get_model_config(model_name)
    targets_from_config = [t.lower() for t in config["targets"]] # type: ignore

    if target_cols is None:
        target_cols = targets_from_config

    # --- 2. Ejecutar Estudio ---
    pbar = tqdm(total=n_trials, file=sys.stderr, desc=f"Optuna ({model_name})", colour="green", nrows=3)  # type: ignore

    """
    tqdm_optuna_callback = TqdmCallback(
        tqdm_object=pbar,
        metric_name="value" # Muestra el 'loss' principal en la barra
    )
    """
    optuna_func = partial(
        optuna_objective, model_name=model_name, target_cols=target_cols, pbar=pbar
    )

    study = optuna.create_study(
        direction="minimize", study_name=f"optuna_{model_name}_{datestudy}_{timestudy}"
    )

    study.optimize(
        optuna_func,
        n_trials=N_TRIALS,
        # callbacks=[tqdm_optuna_callback] # type: ignore
    )

    pbar.close()
    # --- 3. Obtener y Guardar Resultados ---
    best_5 = get_best_trials(study, 5)

    if not best_5:
        print(
            "Optimización de Optuna completada, pero no se encontraron trials exitosos.",
            file=sys.stderr,
        )
        return {}, None, None, None

    optuna_results = PGEN_MODEL_DIR / "optuna_outputs"
    optuna_results.mkdir(parents=True, exist_ok=True)

    filename_base = f"optuna_{model_name}_{datestudy}_{timestudy}"
    filename = Path(optuna_results / filename_base)
    
    _save_optuna_report(study, model_name, optuna_results, filename, best_5)

    # Extraer métricas del mejor trial para el return
    best_trial = best_5[0]
    best_loss = best_trial.value
    best_params = best_trial.params
    normalized_loss = best_trial.user_attrs.get("normalized_loss", 0.0)

    return best_params, best_loss, results_file_json, normalized_loss
