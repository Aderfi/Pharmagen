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

#### Import de librerías estándar ####
import os, sys, json, logging
import pandas as pd
import numpy as np
from pathlib import Path

#### Import de librerías propias ####
import src.config.config as cfg
from src.scripts import *

os.getcwd()
Pharmagen = Path(__file__).parent
sys.path.append(str(Pharmagen))
'''
from src.logger_config import unit_logging
from src.utils import mensaje_introduccion, load_config, check_config
from deepL_model.scripts.train_model import main as train_main
from deepL_model.scripts.predict_model import predict_single_input, predict_from_file
'''

# Configuración de logging
logging.basicConfig(level=logging.INFO)


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