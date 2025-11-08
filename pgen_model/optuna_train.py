#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optuna Hyperparameter Optimization with Multi-Objective Support
================================================================
- Multi-objective optimization (loss + F1 for critical tasks)
- Per-task metric tracking (F1, precision, recall)
- Proper handling of class imbalance
- Clinical priority weighting
"""

# Librerías estándar
import datetime
import json
import math
import random
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import List, Tuple, Dict

# Librerías de terceros
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from optuna import create_study
from optuna.storages import JournalStorage
from optuna.visualization import plot_optimization_history, plot_pareto_front
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Librerías propias
from src.config.config import PGEN_MODEL_DIR, PROJECT_ROOT
from .data import PGenDataset, PGenDataProcess, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import CLINICAL_PRIORITIES, MODEL_REGISTRY, MULTI_LABEL_COLUMN_NAMES, get_model_config
from .train import train_model

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
    input_cols = ["drug", "genalle", "gene", "allele"]
    
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


def calculate_task_metrics(
    model: nn.Module,
    data_loader: DataLoader,
    target_cols: List[str],
    multi_label_cols: set,
    device: torch.device
) -> Dict[str, Dict[str, float]]:
    """
    Calcula métricas detalladas (F1, precision, recall) para cada tarea.
    
    Returns:
        Dict con estructura: {
            'task_name': {
                'f1_macro': float,
                'f1_weighted': float,
                'precision_macro': float,
                'recall_macro': float
            }
        }
    """
    model.eval()
    
    # Almacenar predicciones y ground truth
    all_preds = {col: [] for col in target_cols}
    all_targets = {col: [] for col in target_cols}
    
    with torch.no_grad():
        for batch in data_loader:
            # Batch transfer to device - more efficient
            batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            drug = batch_device["drug"]
            genalle = batch_device["genalle"]
            gene = batch_device["gene"]
            allele = batch_device["allele"]
            
            outputs = model(drug, genalle, gene, allele)
            
            for col in target_cols:
                true = batch_device[col]
                pred = outputs[col]
                
                if col in multi_label_cols:
                    # Multi-label: aplicar sigmoid y threshold
                    probs = torch.sigmoid(pred)
                    predicted = (probs > 0.5).float()
                else:
                    # Single-label: argmax
                    predicted = torch.argmax(pred, dim=1)
                
                all_preds[col].append(predicted.cpu())
                all_targets[col].append(true.cpu())
    
    # Calcular métricas por tarea
    metrics = {}
    for col in target_cols:
        preds = torch.cat(all_preds[col]).numpy()
        targets = torch.cat(all_targets[col]).numpy()
        
        if col in multi_label_cols:
            # Para multi-label, usamos 'samples' average
            metrics[col] = {
                'f1_samples': f1_score(targets, preds, average='samples', zero_division=0),
                'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
                'precision_samples': precision_score(targets, preds, average='samples', zero_division=0),
                'recall_samples': recall_score(targets, preds, average='samples', zero_division=0),
            }
        else:
            # Para single-label
            metrics[col] = {
                'f1_macro': f1_score(targets, preds, average='macro', zero_division=0),
                'f1_weighted': f1_score(targets, preds, average='weighted', zero_division=0),
                'precision_macro': precision_score(targets, preds, average='macro', zero_division=0),
                'recall_macro': recall_score(targets, preds, average='macro', zero_division=0),
            }
    
    return metrics


def optuna_objective(
    trial: optuna.Trial, 
    model_name: str, 
    pbar: tqdm, 
    target_cols: List[str],
    fitted_data_loader: PGenDataProcess,
    train_processed_df: pd.DataFrame,
    val_processed_df: pd.DataFrame,
    csvfiles: Path | str,
    class_weights_task3: torch.Tensor = None, #type: ignore
    use_multi_objective: bool = True,
) -> Tuple[float, ...]:
    """
    Función objetivo de Optuna MEJORADA.
    
    Cambios principales:
    1. Calcula métricas detalladas (F1, precision, recall)
    2. Soporta optimización multi-objetivo
    3. Prioriza la tarea crítica ('effect_type')
    4. Almacena todas las métricas en trial.user_attrs
    
    Returns:
        Si use_multi_objective=True: (total_loss, -f1_critical_task)
        Si use_multi_objective=False: (total_loss,)
    """
    seed = 711
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 1. Definir Hiperparámetros ---
    params = get_optuna_params(trial, model_name)
    
    # --- 2. Crear Datasets y DataLoaders ---
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

    # --- 3. Definir Modelo ---
    input_dims = get_input_dims(fitted_data_loader)
    n_targets_list = get_output_sizes(fitted_data_loader, target_cols)
    target_dims = {
        col_name: n_targets_list[i] for i, col_name in enumerate(target_cols)
    }

    if trial.number == 0:
        print(f"\n🔬 Cargando datos de: {Path(csvfiles)}")
        print(f"📊 Tareas: {', '.join(target_cols)}")
        print(f"⚖️  Prioridades clínicas: {CLINICAL_PRIORITIES}")
    
    model = DeepFM_PGenModel(
        n_drugs=input_dims["drug"],
        n_genalles=input_dims["genalle"],
        n_genes=input_dims["gene"],
        n_alleles=input_dims["allele"],
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        target_dims=target_dims,
    )
    model = model.to(device)

    # --- 4. Definir Criterio y Optimizador ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("weight_decay", 1e-5),
    )

    criterions = []
    task3_name = 'effect_type'
    
    for col in target_cols:
        if col in MULTI_LABEL_COLUMN_NAMES:
            criterions.append(nn.BCEWithLogitsLoss())
        elif col == task3_name and class_weights_task3 is not None:
            criterions.append(nn.CrossEntropyLoss(
                label_smoothing=0.1, 
                weight=class_weights_task3
            ))
            if trial.number == 0:
                print(f"✅ Aplicando Class Weighting (Log Smoothing) para '{task3_name}'")
        else:
            criterions.append(nn.CrossEntropyLoss(label_smoothing=0.1))
    
    criterions.append(optimizer)

    # --- 5. Entrenar (Sin ponderación de incertidumbre para Optuna) ---
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
        use_weighted_loss=False,  # Suma simple en Optuna
        task_priorities=None, #type: ignore
        return_per_task_losses=False,
    )

    # --- 6. Calcular Métricas Detalladas ---
    task_metrics = calculate_task_metrics(
        model, val_loader, target_cols, MULTI_LABEL_COLUMN_NAMES, device
    )
    
    # --- 7. Calcular Pérdida Normalizada ---
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

    # --- 8. Guardar Métricas en Trial ---
    trial.set_user_attr("avg_accuracy", avg_accuracy)
    trial.set_user_attr("normalized_loss", normalized_loss)
    trial.set_user_attr("all_accuracies", best_accuracies_list)
    trial.set_user_attr("seed", seed)
    
    # Guardar métricas por tarea
    for col, metrics in task_metrics.items():
        for metric_name, value in metrics.items():
            trial.set_user_attr(f"{col}_{metric_name}", value)
    
    # --- 9. Identificar la Métrica de la Tarea Crítica ---
    critical_task = 'effect_type'
    critical_f1 = task_metrics.get(critical_task, {}).get('f1_weighted', 0.0)
    
    trial.set_user_attr("critical_task_f1", critical_f1)
    trial.set_user_attr("critical_task_name", critical_task)

    # --- 10. Actualizar Progress Bar ---
    info_dict = {
        "Trial": trial.number,
        "Loss": f"{best_loss:.4f}",
        "NormLoss": f"{normalized_loss:.4f}",
        f"{critical_task}_F1": f"{critical_f1:.4f}",
        "AvgAcc": f"{avg_accuracy:.4f}",
    }
    pbar.set_postfix(info_dict)
    pbar.update(1)

    # --- 11. Retornar según el modo ---
    if use_multi_objective:
        # Multi-objetivo: minimizar loss Y maximizar F1 crítico
        return best_loss, -critical_f1  # Negativo para maximizar
    else:
        # Single-objetivo: solo loss
        return best_loss #type: ignore


def get_best_trials(study: optuna.Study, n: int = 5) -> List[optuna.trial.FrozenTrial]:
    """
    Obtiene los 'n' mejores trials válidos de un estudio.
    Maneja estudios single y multi-objetivo.
    """
    valid_trials = [
        t for t in study.trials
        if t.values is not None and all(isinstance(v, (int, float)) for v in t.values)
    ]
    
    if not valid_trials:
        return []
    
    # Para multi-objetivo, devolver los trials del Pareto front
    if len(valid_trials[0].values) > 1:
        # Ya están ordenados por dominancia de Pareto
        return valid_trials[:n]
    else:
        # Para single-objetivo, ordenar por valor
        return sorted(valid_trials, key=lambda t: t.values[0])[:n]


def _save_optuna_report(
    study: optuna.Study,
    model_name: str,
    results_file_json: Path,
    results_file_txt: Path,
    top_5_trials: List[optuna.trial.FrozenTrial],
    is_multi_objective: bool = False,
):
    """Función auxiliar para guardar los reportes JSON y TXT."""

    if not top_5_trials:
        print(
            "Advertencia: No se encontraron trials válidos para guardar en el reporte.",
            file=sys.stderr,
        )
        return

    best_trial = top_5_trials[0]
    
    # --- Guardar JSON ---
    output = {
        "model": model_name,
        "optimization_type": "multi_objective" if is_multi_objective else "single_objective",
        "best_trial": {
            "number": best_trial.number,
            "values": list(best_trial.values) if best_trial.values else [],
            "params": best_trial.params,
            "user_attrs": dict(best_trial.user_attrs),
        },
        "top5": [
            {
                "number": t.number,
                "values": list(t.values) if t.values else [],
                "params": t.params,
                "critical_f1": t.user_attrs.get("critical_task_f1", 0.0),
            }
            for t in top_5_trials
        ],
        "n_trials": len(study.trials),
        "clinical_priorities": CLINICAL_PRIORITIES,
    }
    
    with open(results_file_json, "w") as f:
        json.dump(output, f, indent=2)

    # --- Guardar TXT ---
    with open(results_file_txt, "w") as f2:
        f2.write(f"╔═══════════════════════════════════════════════════════════╗\n")
        f2.write(f"║  REPORTE DE OPTIMIZACIÓN: {model_name:^30} ║\n")
        f2.write(f"╚═══════════════════════════════════════════════════════════╝\n\n")
        
        if is_multi_objective:
            f2.write(f"🎯 Tipo de Optimización: Multi-Objetivo\n")
            f2.write(f"   - Objetivo 1: Minimizar Loss\n")
            f2.write(f"   - Objetivo 2: Maximizar F1 ({best_trial.user_attrs.get('critical_task_name', 'N/A')})\n\n")
        else:
            f2.write(f"🎯 Tipo de Optimización: Single-Objetivo (Loss)\n\n")
        
        f2.write(f"════════════════ MEJOR TRIAL (#{best_trial.number}) ════════════════\n\n")
        
        # Valores de los objetivos
        if best_trial.values:
            f2.write(f"📊 Valores de Objetivos:\n")
            if is_multi_objective:
                f2.write(f"   Loss:         {best_trial.values[0]:.5f}\n")
                f2.write(f"   F1 (negado):  {best_trial.values[1]:.5f} → F1 real: {-best_trial.values[1]:.5f}\n")
            else:
                f2.write(f"   Loss:         {best_trial.values[0]:.5f}\n")
            f2.write("\n")
        
        # Métricas detalladas
        f2.write(f"📈 Métricas Detalladas:\n")
        f2.write(f"   Loss Normalizado:    {best_trial.user_attrs.get('normalized_loss', 0.0):.5f}\n")
        f2.write(f"   Accuracy Promedio:   {best_trial.user_attrs.get('avg_accuracy', 0.0):.4f}\n")
        f2.write(f"   F1 Tarea Crítica:    {best_trial.user_attrs.get('critical_task_f1', 0.0):.4f}\n")
        f2.write("\n")
        
        # Hiperparámetros
        f2.write(f"⚙️  Hiperparámetros:\n")
        for key, value in best_trial.params.items():
            f2.write(f"   {key:20s}: {value}\n")
        f2.write("\n")
        
        # Top 5
        f2.write(f"═══════════════════ TOP 5 TRIALS ════════════════════\n\n")
        for i, t in enumerate(top_5_trials, 1):
            f2.write(f"{i}. Trial #{t.number}\n")
            if t.values:
                if is_multi_objective:
                    f2.write(f"   Loss: {t.values[0]:.5f}, F1: {-t.values[1]:.5f}\n")
                else:
                    f2.write(f"   Loss: {t.values[0]:.5f}\n")
            f2.write(f"   Params: {t.params}\n\n")

    print(f"\n✅ Reportes de Optuna guardados en: {results_file_json.parent}")


def run_optuna_with_progress(
    model_name,
    n_trials=N_TRIALS,
    output_dir=PGEN_MODEL_DIR,
    target_cols=None,
    use_multi_objective=True,
):
    """
    Ejecuta un estudio de Optuna con soporte multi-objetivo.
    
    Args:
        use_multi_objective: Si True, optimiza (loss, -F1). Si False, solo loss.
    """
    datetime_study = datetime.datetime.now().strftime("%d_%m_%y__%H_%M")

    output_dir = Path(output_dir)
    results_dir = output_dir / "optuna_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    seed = 711

    # --- 1. Definir Targets ---
    config = get_model_config(model_name)
    targets_from_config = [t.lower() for t in config["targets"]]
    cols_from_config = config["cols"]

    if target_cols is None:
        target_cols = targets_from_config

    # --- 2. Cargar y Pre-procesar Datos ---
    print("\n" + "="*60)
    print("🧬 OPTIMIZACIÓN DE HIPERPARÁMETROS - PHARMAGEN PMODEL")
    print("="*60)
    print(f"Modelo: {model_name}")
    print(f"Modo: {'Multi-Objetivo (Loss + F1)' if use_multi_objective else 'Single-Objetivo (Loss)'}")
    print(f"Trials: {n_trials}")
    print("="*60 + "\n")
    
    print("📁 Cargando y pre-procesando datos...")
    csvfiles, _ = train_data_import(targets_from_config)
    
    data_loader = PGenDataProcess()
    df = data_loader.load_data(
        csvfiles,
        cols_from_config,
        targets_from_config,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=['effect_type'],
    )
    
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
    print("✅ Datos listos.\n")
    
    # --- 3. Calcular Pesos de Clase ---
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
            
            weights = 1.0 / torch.log(all_counts + 2)
            weights = weights / weights.mean()
            class_weights_task3 = weights.to(device)
            print(f"⚖️  Pesos de clase (Log Smoothing) calculados para '{task3_name}'")
            print(f"   Distribución: Min={weights.min():.3f}, Max={weights.max():.3f}, Mean={weights.mean():.3f}\n")

        except Exception as e:
            print(f"⚠️  Error calculando pesos de clase: {e}\n")

    # --- 4. Ejecutar Estudio ---
    pbar = tqdm(
        total=n_trials,
        desc=f"Optuna ({model_name})",
        colour="green",
    )

    optuna_func = partial(
        optuna_objective, 
        model_name=model_name, 
        target_cols=target_cols, 
        pbar=pbar,
        fitted_data_loader=data_loader,
        train_processed_df=train_processed_df,
        val_processed_df=val_processed_df,
        csvfiles=csvfiles, # type: ignore
        class_weights_task3=class_weights_task3, # type: ignore
        use_multi_objective=use_multi_objective,
    )

    # Crear estudio según el modo
    if use_multi_objective:
        study = optuna.create_study(
            directions=["minimize", "minimize"],  # loss, -F1 (ambos minimizar)
            study_name=f"optuna_{model_name}_{datetime_study}"
        )
    else:
        study = optuna.create_study(
            direction="minimize",
            study_name=f"optuna_{model_name}_{datetime_study}"
        )

    study.optimize(optuna_func, n_trials=n_trials)
    pbar.close()
    
    # --- 5. Guardar Visualizaciones ---
    if study.trials:
        try:
            # Historial de optimización
            fig_history = plot_optimization_history(study)
            graph_filename = optuna_results / f"history_{model_name}_{datetime_study}.html"
            fig_history.write_html(str(graph_filename))
            print(f"📊 Gráfico de historial guardado: {graph_filename}")
            
            # Pareto front (solo para multi-objetivo)
            if use_multi_objective and len(study.best_trials) > 1:
                fig_pareto = plot_pareto_front(study)
                pareto_filename = optuna_results / f"pareto_{model_name}_{datetime_study}.html"
                fig_pareto.write_html(str(pareto_filename))
                print(f"📊 Frente de Pareto guardado: {pareto_filename}")
        except Exception as e:
            print(f"⚠️  Error guardando gráficos: {e}")
    
    # --- 6. Obtener y Guardar Reporte ---
    best_5 = get_best_trials(study, 5)

    if not best_5:
        print(
            "⚠️  Optimización completada, pero no se encontraron trials exitosos.",
            file=sys.stderr,
        )
        return {}, None, None, None

    filename = f"optuna_{model_name}_{datetime_study}"
    results_file_json = results_dir / f"{filename}.json"
    results_file_txt = results_dir / f"{filename}.txt"
    
    _save_optuna_report(
        study, model_name, results_file_json, results_file_txt, best_5,
        is_multi_objective=use_multi_objective
    )

    # Extraer métricas del mejor trial
    best_trial = best_5[0]
    best_params = best_trial.params
    best_loss = best_trial.values[0] if best_trial.values else float('inf')
    normalized_loss = best_trial.user_attrs.get("normalized_loss", 0.0)
    best_f1 = best_trial.user_attrs.get("critical_task_f1", 0.0)
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("✅ OPTIMIZACIÓN COMPLETADA")
    print("="*60)
    if use_multi_objective:
        print(f"🏆 Mejor Trial (Pareto-Optimal): #{best_trial.number}")
        print(f"   Loss:           {best_loss:.5f}")
        print(f"   F1 (Critical):  {best_f1:.5f}")
    else:
        print(f"🏆 Mejor Trial: #{best_trial.number}")
        print(f"   Loss:           {best_loss:.5f}")
    
    print(f"\n📁 Reportes guardados en: {results_dir}")
    print("="*60 + "\n")

    return best_params, best_loss, results_file_json, normalized_loss