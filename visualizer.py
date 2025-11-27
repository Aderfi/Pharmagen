import logging
import datetime
import sys
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import optuna
from optuna import Trial, Study, visualization
from optuna.study import StudySummary as OptunaStudySummary

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# --- IMPORTACIONES DEL PROYECTO ---
# Asumimos que la estructura de carpetas es src/model.py, src/data.py, etc.
from src.model import DeepFM_PGenModel
from src.data import PGenDataLoader, PGenDataset
from src.losses import FocalLoss, MultiTaskUncertaintyLoss 
from src.cfg.config import MODELS_DIR
from src.utils.data import load_and_prep_dataset
from src.train import train_model  # Importamos la función de entrenamiento

# Configuración de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MULTI_LABEL_COLS = ["phenotype_outcome"]

# ==============================================================================
# 1. FUNCIÓN DE CARGA Y PREPARACIÓN
# ==============================================================================
def load_data_and_process():
    """
    Carga datos, normaliza columnas y prepara los pesos de las clases.
    """
    # Ajusta esta ruta si tu archivo está en otro lugar
    DATA_PATH = "train_data/final_enriched_data.tsv"
    
    if not Path(DATA_PATH).exists():
        logger.error(f"❌ No se encontró el dataset en {DATA_PATH}")
        sys.exit(1)

    logger.info("📂 Cargando dataset...")
    
    # Definición de columnas esperadas
    feature_cols = [
        "atc", "drug", "gene_symbol", "variant_normalized", 
        "variant/haplotypes", "previous_condition_term"
    ]
    target_main = "phenotype_outcome"
    # Definimos las 3 tareas objetivos
    target_cols = [target_main, "effect_direction", "effect_type"]
    
    # Usamos load_and_prep_dataset de src.utils.data
    # Nota: all_cols debe incluir features + targets
    all_cols = feature_cols + target_cols
    
    df = load_and_prep_dataset(
        csv_path=DATA_PATH, 
        all_cols=all_cols, 
        target_cols=target_cols, 
        multi_label_targets=["phenotype_outcome"], # Asumimos targets simples para este script de evaluación
        stratify_cols = "phenotype_outcome"
    )
    
    # --- FIX 1: NORMALIZACIÓN DE COLUMNAS ---
    # load_and_prep_dataset ya hace lower(), pero aseguramos limpieza extra
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Verificar existencia de targets después de la carga
    missing_targets = [t for t in target_cols if t not in df.columns]
    if missing_targets:
        logger.error(f"❌ Faltan columnas objetivo críticas: {missing_targets}")
        sys.exit(1)
    
    # Split Estratificado
    logger.info("✂️ Dividiendo Train/Test...")
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=711, 
        stratify=df["stratify_col"]
    )
    
    # Procesador
    logger.info("⚙️ Ajustando Encoders...")
    processor = PGenDataLoader(feature_cols=feature_cols, target_cols=target_cols, multi_label_cols=["phenotype_outcome"])
    processor.fit(train_df)
    
    # --- CÁLCULO DE ALPHA (CLASS WEIGHTS) ---
    alpha_tensor = None
    if target_main in df.columns:
        logger.info(f"⚖️ Calculando pesos para matriz MultiLabel: {target_main}")
        encoder = processor.encoders[target_main]
        temp_df = train_df[[target_main]].copy()
        try:
            y_encoded = encoder.transform(temp_df[target_main].apply(processor._split_labels))
        except:
            full_encoded = processor.transform(temp_df)
            y_series = full_encoded[target_main]
            y_encoded = np.vstack(y_series.values) # type: ignore
        
        n_samples = y_encoded.shape[0]
        n_classes = y_encoded.shape[1]

        class_counts = np.sum(y_encoded, axis=0)
        class_counts = np.maximum(class_counts, 1)
        weights = n_samples / (n_classes * class_counts)
        alpha_tensor = torch.tensor(weights, dtype=torch.float32)
        logger.info(f"   Shape matriz targets: {y_encoded.shape}")
        logger.info(f"✅ Pesos calculados manualmente: {alpha_tensor.tolist()}")
    
    return train_df, val_df, processor, feature_cols, target_cols, alpha_tensor

# ==============================================================================
# 2. LÓGICA PRINCIPAL
# ==============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Iniciando evaluación en: {device}")

    # Parámetros del Trial 39
    best_params = {
        "embedding_dim": 512,
        "n_layers": 5,
        "hidden_dim": 512,
        "activation_function": "gelu",
        "learning_rate": 0.0017328736719051743,
        "batch_size": 32,
        "optimizer_type": "adamw",
        "dropout_rate": 0.5388290991820363,
        "embedding_dropout": 0.29179632489341656,
        "weight_decay": 2.7935421438995612e-05,
        "label_smoothing": 0.0003925552093576529,
        "use_batch_norm": True,
        "use_layer_norm": False,
        "focal_gamma": 5.975709827046388,
        "manual_task_weights": False,
        "use_uncertainty_loss": True,
        "fm_dropout": 0.1,
        "fm_hidden_layers": 0
    }

    # Carga de datos
    train_df, test_df, processor, feature_cols, target_cols, alpha = load_data_and_process()
    
    logger.info("🔄 Transformando dataframes...")
    train_processed = processor.transform(train_df)
    test_processed = processor.transform(test_df)
    
    # Mapeo de features
    n_features_map = {}
    valid_feature_cols = []
    for col in feature_cols:
        if col in processor.encoders:
            n_features_map[col] = len(processor.encoders[col].classes_)
            valid_feature_cols.append(col)
        else:
            logger.warning(f"⚠️ Feature '{col}' ignorada.")

    if not n_features_map:
        sys.exit(1)

    # Mapeo de targets
    target_dims = {}
    for t in target_cols:
        if t in processor.encoders:
            # Si es MultiLabelBinarizer, classes_ es la lista de etiquetas únicas
            target_dims[t] = len(processor.encoders[t].classes_)
        else:
            target_dims[t] = 1

    logger.info(f"🧠 Inicializando Modelo. Dims Targets: {target_dims}")
    model = DeepFM_PGenModel(
        n_features=n_features_map,
        target_dims=target_dims,
        embedding_dim=best_params["embedding_dim"],
        hidden_dim=best_params["hidden_dim"],
        dropout_rate=best_params["dropout_rate"],
        n_layers=best_params["n_layers"],
        activation_function=best_params["activation_function"],
        embedding_dropout=best_params["embedding_dropout"],
        fm_dropout=best_params["fm_dropout"],
        use_batch_norm=best_params["use_batch_norm"],
        use_layer_norm=best_params["use_layer_norm"],
        fm_hidden_layers=best_params["fm_hidden_layers"]
    )
    model.to(device)

    # Criterios y Optimizador
    criterions = []
    target_main = target_cols[0] # "phenotype_outcome"

    for col in target_cols:
        if col == target_main:
            logger.info(f"   - Loss para {col}: FocalLoss (Multilabel support)")
            focal_loss = FocalLoss(
                alpha=alpha.to(device) if alpha is not None else None, 
                gamma=best_params["focal_gamma"],
                label_smoothing=best_params["label_smoothing"]
            )
            criterions.append(focal_loss)
        else:
            logger.info(f"   - Loss para {col}: CrossEntropyLoss")
            ce_loss = nn.CrossEntropyLoss(label_smoothing=best_params["label_smoothing"])
            criterions.append(ce_loss)
    
    # Uncertainty Loss
    uncertainty_module = None
    model_params = list(model.parameters())
    if best_params["use_uncertainty_loss"]:
        logger.info("🔧 Activando MultiTaskUncertaintyLoss")
        uncertainty_module = MultiTaskUncertaintyLoss(task_names=target_cols).to(device)
        model_params += list(uncertainty_module.parameters())
    
    optimizer = torch.optim.AdamW(
        model_params,
        lr=best_params["learning_rate"], 
        weight_decay=best_params["weight_decay"]
    )
    criterions.append(optimizer)

    # Entrenamiento
    logger.info("🔄 Re-entrenando...")
    
    # === CORRECCIÓN AQUÍ ===
    # Pasamos multi_label_cols={target_main} para que PGenDataset sepa que es una matriz
    target_main_set = {target_main} 
    
    train_dataset = PGenDataset(
        train_processed, 
        valid_feature_cols, 
        target_cols, 
        multi_label_cols=target_main_set # <--- ESTO SOLUCIONA TU ERROR
    )
    val_dataset = PGenDataset(
        test_processed, 
        valid_feature_cols, 
        target_cols, 
        multi_label_cols=target_main_set # <--- ESTO TAMBIÉN
    )
    
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)
    
    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        criterions=criterions,
        epochs=15, 
        patience=5,
        model_name="reconstructed_best_trial",
        feature_cols=valid_feature_cols,
        target_cols=target_cols,
        device=device,
        progress_bar=True,
        uncertainty_loss_module=uncertainty_module,
        multi_label_cols=target_main_set # Importante pasarlo aquí también
    )

    # Evaluación
    logger.info("\n📊 Evaluando métricas...")
    model.eval()
    
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k in valid_feature_cols}
            labels = batch[target_main].to(device) 
            
            outputs = model(inputs)
            logits = outputs[target_main]
            
            probs = torch.sigmoid(logits) 
            
            all_preds.append(probs.cpu())
            all_targets.append(labels.cpu())

    y_true = torch.cat(all_targets).numpy()
    y_prob = torch.cat(all_preds).numpy()
    
    logger.info("Generando reporte Multilabel (Threshold 0.5)...")
    y_pred = (y_prob > 0.5).astype(int)
    
    class_names = list(processor.encoders[target_main].classes_)
    
    print("\n" + "="*60)
    print(f" REPORTE CLÍNICO FINAL: {target_main}")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

def plot_custom_history_matplotlib(study, target_name="Loss"):
    """
    Genera el gráfico de historia de optimización usando Matplotlib
    con puntos individuales, media móvil y regresión polinómica.
    """
    # Extraer datos válidos (evitar trials fallidos o sin valor)
    trials_validos = [t for t in study.trials if t.values is not None]
    
    if not trials_validos:
        print("No hay trials válidos para visualizar.")
        return

    # Asumiendo que queremos el primer objetivo (values[0])
    x = np.array([t.number for t in trials_validos])
    y = np.array([t.values[0] for t in trials_validos])

    # Crear figura
    fit_mat = plt.figure(figsize=(10, 6))
    ax = fit_mat.add_subplot(111)

    # 1. Trials Individuales
    ax.plot(x, y, marker='o', linestyle='', color='red', alpha=0.3, label='Trials Individuales')

    # 2. Añadir Media Móvil (Tendencia Local)
    # Ventana dinámica: mínimo 5, o 10% de los datos si es mayor
    window = max(int(len(x)/10), 5)
    y_smooth = pd.Series(y).rolling(window=window).mean()
    #ax.plot(x, y_smooth, color='blue', linewidth=2.5, label=f'Media Móvil (n={window})')

    # 3. Añadir Regresión Polinómica (Tendencia Global)
    if len(x) > 2:
        
        z = np.polyfit(np.log(x + 1), y, 1) 
        # p[0] es la pendiente (a), p[1] es la intersección (b)
        ax.plot(x, z[0] * np.log(x + 1) + z[1], "k--", linewidth=1.5, label='Tendencia Logarítmica')
        """    
        
        lowess = sm.nonparametric.lowess(y, x, frac=0.3)
        lowess_x = lowess[:, 0]
        lowess_y = lowess[:, 1]
        ax.plot(lowess_x, lowess_y, "k--", linewidth=5, label='Tendencia LOWESS')
        """"""
        try:
            z = np.polyfit(x, y, 3)
            p = np.poly1d(z)
            ax.plot(x, p(x), "k--", linewidth=1.5, label='Tendencia Global (Poly d=2)')
        except Exception as e:
            print(f"No se pudo calcular la regresión polinómica: {e}")
        """
    # Decoración
    ax.set_title(f'Historia de Optimización: {study.study_name}')
    ax.set_xlabel('Número de Trial')
    ax.set_ylabel(target_name)
    ax.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Mostrar el gráfico de Matplotlib
    print("Mostrando gráfico de historia (Matplotlib)...")
    plt.show()

def load_study_from_db(study_name, storage_url):

    study = optuna.load_study(study_name=study_name, storage=storage_url)
        
    return study, False
     

"""  
if __name__ == "__main__":
    main()
"""
if __name__ == "__main__":
    raw_name = "OPT_Phenotype_Effect_Outcome_2025_11_24_11_22.db"

    study_path = Path("reports/optuna_reports/study_DBs/", raw_name)

    study_name = raw_name.replace(".db", "")
    storage_url = f"sqlite:///{study_path.resolve()}"

    study, multi = load_study_from_db(study_name=study_name, storage_url=storage_url)
    print(f"Estudio '{study_name}' cargado exitosamente.")
    
    direction = study.direction
    best_trial = study.best_trial
    user_attributes = study.user_attrs
    system_attributes = study.system_attrs
    ttdt = "2025_11_24_11_22"
    datetime_study = datetime.datetime.strptime(ttdt, "%Y_%m_%d_%H_%M")
    study_identifier = study._study_id

    #print(f"\nMejor Trial ID: {best_trial.number}")
    #print(f"Valor Objetivo (Loss): {best_trial.value}")
    #print("Mejores Hiperparámetros recuperados:")
    #for key, value in best_trial.params.items():
    #    print(f"  - {key}: {value}")
    
    
    #print(study.best_trial)
