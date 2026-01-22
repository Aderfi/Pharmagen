# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# coding=utf-8
"""
Pipeline de entrenamiento para modelos farmacogenéticos.

Orquesta el flujo completo:
1. Carga de configuración (TOML).
2. Procesamiento de datos (PGenDataProcess).
3. Instanciación del modelo (DeepFM).
4. Configuración de Estrategia de Loss (Focal / Uncertainty).
5. Bucle de Entrenamiento.
6. Persistencia de artefactos.
"""

import logging
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# --- Imports del Proyecto ---
from src.cfg.config import MODEL_ENCODERS_DIR, MODELS_DIR, MULTI_LABEL_COLUMN_NAMES #noqa
from src.cfg.model_configs import get_model_config
from src.data import PGenDataProcess, PGenDataset
from src.loss_functions import MultiTaskUncertaintyLoss
from src.model import DeepFM_PGenModel
from src.train import save_model, train_model
from src.utils.data import load_and_prep_dataset
from src.utils.training import (
    create_optimizer,
    create_scheduler,
    create_task_criterions,
)

logger = logging.getLogger(__name__)


def train_pipeline(
    csv_path: Path,
    model_name: str,
    target_cols: list[str] | None = None,
    patience: int | None = None,
    epochs: int | None = None,
    random_seed: int = 711
) -> None:
    """
    Ejecuta el pipeline de entrenamiento end-to-end.

    Args:
        csv_path: Ruta al dataset fuente.
        model_name: Clave del modelo en models.toml.
        target_cols: Sobreescritura opcional de targets (usualmente None).
        patience: Sobreescritura opcional de early stopping.
        epochs: Sobreescritura opcional de épocas máximas.
        random_seed: Semilla para reproducibilidad.
    """
    # 1. Reproducibilidad
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Iniciando pipeline para: {model_name} en {device}")

    # 2. Carga de Configuración
    # Fusiona defaults + models.toml
    config = get_model_config(model_name)
    params = config["params"]

    # Sobreescrituras de argumentos de función (prioridad máxima)
    final_epochs = epochs or config.get("epochs", 100)
    final_patience = patience or params.get("early_stopping_patience", 25)
    
    feature_cols = [c.lower() for c in config["features"]]
    # Si no se pasan targets manuales, usar los del config
    eff_target_cols = [t.lower() for t in (target_cols or config["targets"])]

    # 3. Procesamiento de Datos
    logger.info("Cargando y procesando datos...")
    processor = PGenDataProcess(
        feature_cols=feature_cols,
        target_cols=eff_target_cols,
        multi_label_cols=list(MULTI_LABEL_COLUMN_NAMES)
    )
    
    # Lista total de columnas a cargar
    cols_to_load = list(set(config["cols"] + config.get("features", []) + config.get("targets", [])))

    df = load_and_prep_dataset(
        csv_path=csv_path,
        all_cols=cols_to_load,
        target_cols=eff_target_cols,
        multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
        stratify_cols=config.get("stratify"),
    )

    # Split Estratificado
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df.get("stratify_col"), # Usa .get por si no se generó estratificación
        random_state=random_seed
    )

    # Fit & Transform
    processor.fit(train_df)
    train_enc = processor.transform(train_df)
    val_enc = processor.transform(val_df)

    # Datasets y Loaders
    # Pin_memory acelera la transferencia CPU -> GPU
    batch_size = params["batch_size"]
    train_ds = PGenDataset(train_enc, feature_cols, eff_target_cols, MULTI_LABEL_COLUMN_NAMES)
    val_ds = PGenDataset(val_enc, feature_cols, eff_target_cols, MULTI_LABEL_COLUMN_NAMES)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # 4. Inicialización del Modelo
    # Calculamos dimensiones dinámicamente basadas en los encoders ajustados
    n_features_dims = {col: len(enc.classes_) for col, enc in processor.encoders.items() if col in feature_cols}
    target_dims = {col: len(enc.classes_) for col, enc in processor.encoders.items() if col in eff_target_cols}

    logger.info("Inicializando arquitectura DeepFM...")
    model = DeepFM_PGenModel(
        n_features=n_features_dims,
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

    # 5. Configuración de Entrenamiento (Loss & Optimizer)
    # A. Funciones de pérdida individuales
    loss_fns_dict = create_task_criterions(
        eff_target_cols, MULTI_LABEL_COLUMN_NAMES, params, device
    )

    # B. Módulo de Incertidumbre (Opcional)
    uncertainty_module = None
    if params.get("use_uncertainty_loss", False):
        logger.info("Activando MultiTaskUncertaintyLoss (Kendall & Gal)")
        uncertainty_module = MultiTaskUncertaintyLoss(eff_target_cols).to(device)

    # C. Optimizador (debe conocer los params del modelo Y de la incertidumbre)
    optimizer = create_optimizer(model, params, uncertainty_module)

    # D. Scheduler
    scheduler = create_scheduler(optimizer, params)

    # E. Preparar lista para legacy train_model: [Loss1, Loss2, ..., Optimizer]
    # IMPORTANTE: El orden debe coincidir exactamente con eff_target_cols
    criterions_list: list[Any] = [loss_fns_dict[col] for col in eff_target_cols]
    criterions_list.append(optimizer)

    # F. Pesos manuales (si no se usa incertidumbre automática)
    task_priorities = config.get("manual_task_weights_dict", None) if config.get("manual_task_weights") else None

    # 6. Bucle de Entrenamiento
    logger.info(f"Comenzando entrenamiento por {final_epochs} épocas...")
    
    best_loss, best_acc, task_losses = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterions=criterions_list,
        epochs=final_epochs,
        patience=final_patience,
        model_name=model_name,
        feature_cols=feature_cols,
        target_cols=eff_target_cols,
        device=device,
        scheduler=scheduler,
        multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
        task_priorities=task_priorities,
        return_per_task_losses=True,
        progress_bar=True, # Aquí sí queremos ver la barra
        uncertainty_loss_module=uncertainty_module, # Kwarg mágico que maneja train_model refactorizado
    )

    # 7. Persistencia
    logger.info("Guardando artefactos...")
    
    # Guardar Modelo y Reporte
    save_model(
        model=model,
        target_cols=eff_target_cols,
        best_loss=best_loss,
        best_accuracies=best_acc,
        model_name=model_name,
        avg_per_task_losses=task_losses,
        params_to_txt=params
    )

    # Guardar Encoders (Crítico para inferencia)
    # Aseguramos que el directorio existe (gracias a paths.toml ya debería, pero por seguridad)
    Path(MODEL_ENCODERS_DIR).mkdir(parents=True, exist_ok=True)
    enc_path = Path(MODEL_ENCODERS_DIR) / f"encoders_{model_name}.pkl"
    joblib.dump(processor.encoders, enc_path)
    
    logger.info(f"Pipeline finalizado exitosamente. Encoders en: {enc_path}")

if __name__ == "__main__":
    # Bloque de prueba simple
    pass