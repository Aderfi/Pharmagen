# Copyright (C) 2023 [Tu Nombre / Pharmagen Team]
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

# src/interface/cli.py
# Control de interfaz de usuario
import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Proyecto
from src.config.config import DATA_DIR, DATE_STAMP
from src.config.model_configs import select_model
from src.pgen_model.pipeline import train_pipeline
from src.pgen_model.optuna_tuner import run_optuna_study
from src.pgen_model.predict import PGenPredictor

# UI
from src.interface.utils import Spinner, input_path, print_header, print_success, print_error

logger = logging.getLogger(__name__)

# ==============================================================================
# FLUJOS DE TRABAJO (Workflows)
# ==============================================================================

def run_genomic_processing():
    """Simulación del flujo de ETL genómico."""
    print_header("Módulo de Procesamiento Genómico")
    logger.info("Iniciando módulo genómico interactivo.")
    
    # Aquí iría la llamada real a src/data_handle/...
    with Spinner("Analizando archivos VCF y mapeando variantes..."):
        time.sleep(2) # Simulación
        
    print_success("Procesamiento completado (Simulado).")


def run_training_flow():
    """Flujo interactivo para entrenar modelos."""
    print_header("Módulo de Entrenamiento")
    
    # 1. Selección de Modelo
    model_name = select_model("Selecciona el modelo a entrenar:")
    
    # 2. Selección de Datos
    # Sugerimos una ruta por defecto si existe
    default_data = DATA_DIR / "processed" / "training_data.tsv"
    if not default_data.exists(): default_data = None
    
    csv_path = input_path("Ruta del archivo de entrenamiento (CSV/TSV)", default=default_data)

    # 3. Selección de Modo
    print("\nModo de entrenamiento:")
    print("  1. Entrenamiento Estándar (Un solo ciclo)")
    print("  2. Optimización de Hiperparámetros (Optuna)")
    
    mode = input("Selecciona (1-2): ").strip()
    
    if mode == "1":
        _run_standard_training(model_name, csv_path)
    elif mode == "2":
        _run_optuna_training(model_name, csv_path)
    else:
        print_error("Opción inválida.")

def _run_standard_training(model_name: str, csv_path: Path):
    epochs_str = input("Número de épocas [100]: ").strip()
    epochs = int(epochs_str) if epochs_str.isdigit() else 100
    
    print(f"\nIniciando entrenamiento estándar para '{model_name}'...")
    with Spinner("Configurando pipeline y cargando datos..."):
        # La configuración es rápida, el entrenamiento real mostrará su propia barra
        pass
        
    train_pipeline(csv_path=csv_path, model_name=model_name, epochs=epochs)
    print_success("Entrenamiento finalizado.")

def _run_optuna_training(model_name: str, csv_path: Path):
    trials_str = input("Número de trials [50]: ").strip()
    n_trials = int(trials_str) if trials_str.isdigit() else 50
    
    print(f"\nIniciando Optuna para '{model_name}' ({n_trials} trials)...")
    # Optuna manejará su propia barra de progreso
    run_optuna_study(model_name=model_name, csv_path=csv_path, n_trials=n_trials)
    print_success("Estudio de optimización finalizado.")


def run_prediction_flow():
    """Flujo interactivo para inferencia."""
    print_header("Módulo de Predicción")
    
    model_name = select_model("Selecciona el modelo para predecir:")
    
    try:
        # Instancia única del predictor (Singleton-like scope)
        with Spinner("Cargando modelo y encoders en memoria..."):
            predictor = PGenPredictor(model_name)
        print_success("Modelo cargado correctamente.")

        while True:
            print("\n--- Menú Predicción ---")
            print("  1. Predicción Interactiva (Single)")
            print("  2. Predicción por Lotes (Archivo)")
            print("  3. Volver al menú principal")
            
            sub_choice = input("Opción: ").strip()

            if sub_choice == "1":
                _interactive_predict_loop(predictor)
            elif sub_choice == "2":
                _batch_predict_flow(predictor)
            elif sub_choice == "3":
                break
                
    except FileNotFoundError as e:
        logger.error(f"Error cargando modelo: {e}")
        print_error(f"No se encontró el modelo o encoders: {e}")
        print("Tip: Entrena el modelo primero.")
    except Exception as e:
        logger.error(f"Error crítico en predicción: {e}", exc_info=True)
        print_error(f"Error inesperado: {e}")


def _interactive_predict_loop(predictor: PGenPredictor):
    print("\n(Escribe 'q' para cancelar en cualquier momento)")
    inputs = {}
    
    # Solicitud dinámica basada en los features que el modelo necesita
    for feature in predictor.feature_cols:
        val = input(f"Ingrese valor para '{feature}': ").strip()
        if val.lower() == 'q': return
        inputs[feature] = val
    
    print("\nCalculando...")
    result = predictor.predict_single(inputs)
    
    print("\n--- Resultados ---")
    if result:
        for k, v in result.items():
            print(f"  🔹 {k}: {v}")
    else:
        print_error("Error en la predicción.")


def _batch_predict_flow(predictor: PGenPredictor):
    path = input_path("Ruta del archivo CSV/TSV de entrada")
    
    with Spinner(f"Procesando {path.name}..."):
        results = predictor.predict_file(path)
    
    if not results:
        print("⚠️ No se generaron resultados.")
        return

    out_path = path.parent / f"{path.stem}_predictions_{DATE_STAMP}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print_success(f"Predicciones guardadas en: {out_path}")

def run_advanced_analysis():
    print_header("Análisis Avanzado")
    print("Generando reportes de interpretabilidad...")
    print("Funcionalidad en construcción.")

# ==============================================================================
# MENÚ PRINCIPAL LOOP
# ==============================================================================

def main_menu_loop():
    logger.info("Iniciando menú interactivo.")
    while True:
        print_header(f"Pharmagen v0.667 - Menú Principal")
        print("  1. Procesar Datos Genómicos (ETL)")
        print("  2. Entrenar Modelos (Deep Learning)")
        print("  3. Realizar Predicciones (Inferencia)")
        print("  4. Análisis Avanzado")
        print("  5. Salir")
        print("="*60)
        
        choice = input("Selecciona opción (1-5): ").strip()
        
        if choice == "1":
            run_genomic_processing()
        elif choice == "2":
            run_training_flow()
        elif choice == "3":
            run_prediction_flow()
        elif choice == "4":
            run_advanced_analysis()
        elif choice == "5":
            logger.info("Salida del sistema por el usuario.")
            print("\n¡Hasta luego!")
            sys.exit(0)
        else:
            print("Opción no válida.")