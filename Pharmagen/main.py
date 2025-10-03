"""
#  Software: pharmagen_pmodel
# Versión: 0.1
# Autor: Astordna / Aderfi / Adrim Hamed Outmani
# Fecha: 2024-06-15
# Descripción: Este software tiene la utilidad de actuar como puente para los inputs y la interpretación
#              de los outputs de un modelo predictivo cuya finalidad es inferir a partir de datos genómicos
#              del paciente, la eficacia terapéutica y el posible riesgo incrementado de ciertas toxicidades.
#              El modelo predictivo está basado en un conjunto de modelos de machine learning entrenados
#              con datos genómicos y clínicos de pacientes reales.
"""

import sys
import src as src
import os
from pathlib import Path

# --- Importación de paths y metadatos desde config.py ---
from .config import PHARMAGEN_DIR, LOGS_DIR, CACHE_DIR, SRC_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, DOCS_DIR
   
# Si tienes rutas a deepL_model/scripts, añádelas aquí

# Si necesitas rutas específicas a scripts de deepL_model:
MODEL_SCRIPTS_DIR = Path("deepL_model") / "scripts"

# --- Añadir rutas a sys.path para importación de módulos internos ---
sys.path.append(str(PHARMAGEN_DIR))
from src.logger_config import unit_logging
from src.utils import mensaje_introduccion, load_config, check_config
from deepL_model.scripts.train_model import main as train_main
from deepL_model.scripts.predict_model import predict_single_input, predict_from_file

def main():
    # 1. Inicializa logging y muestra introducción
    unit_logging()
    print(mensaje_introduccion())
    config_df = load_config()

    while True:
        print("\n¿Qué deseas hacer?")
        print("1. Entrenar modelo")
        print("2. Realizar predicción (introducir datos manualmente)")
        print("3. Realizar predicción (desde archivo)")
        print("4. Salir")

        choice = input("Introduce 1, 2, 3 o 4: ").strip()
        if choice not in {'1', '2', '3', '4'}:
            print("Opción no válida. Inténtalo de nuevo.")
            continue

        if choice == "1":
            print("\nIniciando flujo de entrenamiento...")
            train_main()
        elif choice == "2":
            print("\nIntroduce los datos del paciente para predicción:")
            mutaciones = input("Mutaciones (separadas por coma): ")
            medicamentos = input("Medicamentos (separados por coma): ")
            resultado = predict_single_input(mutaciones, medicamentos)
            print("\nResultado de la predicción:")
            for k, v in resultado.items():
                print(f"{k}: {v}")
        elif choice == "3":
            file_path = input("\nIntroduce la ruta del archivo CSV: ")
            try:
                results = predict_from_file(file_path)
                print("\nResultados de la predicción para cada paciente:")
                print(results)
            except Exception as e:
                print(f"Error al procesar el archivo: {e}")
        elif choice == "4":
            print("\n¡Gracias por usar Pharmagen!")
            sys.exit(0)

if __name__ == "__main__":
    main()