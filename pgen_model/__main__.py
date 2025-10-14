"""
Punto de entrada para el paquete pgen_model.
Proporciona un menú CLI para entrenar y utilizar el modelo predictivo.
"""
import math
import json
import sys

from pathlib import Path

from .metrics import metrics_models
from .pipeline import train_pipeline
from .model_configs import MODEL_CONFIGS
from .predict import predict_single_input, predict_from_file
from .optuna_train import run_optuna_with_progress
from .data import PGenInputDataset, train_data_load

###########################################################
# Configuración de Optuna
N_TRIALS = 100  # Número de pruebas para Optuna
PGEN_MODEL_DIR = "."  # Ajusta si tu variable global es distinta
###########################################################

def select_model(model_options, prompt="Selecciona el modelo:"):
    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(model_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    model_choice = ""
    while model_choice not in [str(i+1) for i in range(len(model_options))]:
        model_choice = input(f"{prompt} (1-{len(model_options)}): ").strip()
        if model_choice not in [str(i+1) for i in range(len(model_options))]:
            print("Opción no válida. Intente de nuevo.")
    return model_options[int(model_choice)-1]

def main():
    model_options = list(MODEL_CONFIGS.keys())

    while True:
        print("""
 ———————————————— Pharmagen PModel ————————————————
| 1. Entrenar modelo                               |
| 2. Realizar predicción (datos manuales)          |
| 3. Realizar predicción (desde archivo)           |
| 4. Optimizacion de hiperparámetros (Optuna)      |
| 5. Salir                                         |
 ——————————————————————————————————————————————————
""")
        choice = input("Selecciona opción (1-5): ").strip()

        if choice == "1":
            model_name = select_model(model_options, "Selecciona el modelo para entrenamiento")
            batch_size, epochs, lr, emb_dim, hidden_dim, dropout_rate, patience = metrics_models(model_name) # type: ignore
            params = {
                'emb_dim': emb_dim,
                'hidden_dim': hidden_dim,
                'dropout_rate': dropout_rate,
                'learning_rate': lr,
                'patience': patience,
                'batch_size': batch_size,
                'epochs': epochs
            }
            PMODEL_DIR = "pgen_model/models/"
            csv_files = ["train.csv"]
            print(f"Iniciando entrenamiento con modelo: {model_name}")
            train_pipeline(PMODEL_DIR, csv_files, model_name, params, epochs=epochs, patience=patience, batch_size=batch_size)

        elif choice == "2":
            print("Introduce datos del paciente para predicción:")
            mutaciones = input("Mutaciones (separadas por coma): ")
            medicamentos = input("Medicamentos (separados por coma): ")
            resultado = predict_single_input(mutaciones, medicamentos)
            print("\nResultado de la predicción:")
            if resultado is not None:
                for k, v in resultado.items():  # type: ignore
                    print(f"{k}: {v}")
            else:
                print("No se pudo realizar la predicción.")

        elif choice == "3":
            file_path = input("Ruta del archivo CSV: ")
            try:
                resultados = predict_from_file(file_path)
                print("\nResultados de las predicciones:")
                print(resultados)
            except Exception as e:
                print(f"Error procesando archivo: {e}")

        elif choice == "4":
            import optuna
            # Silenciar logs de Optuna salvo errores
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            print("\nOptimizando hiperparámetros con Optuna...")
            optuna_model_name = select_model(model_options, "Selecciona el modelo para optimización")
            best_params, best_loss, results_file, normalized_loss = run_optuna_with_progress(
                optuna_model_name,
                n_trials=N_TRIALS,
                output_dir=Path(PGEN_MODEL_DIR)
            )
            
            print(f"\nMejores hiperparámetros encontrados ({optuna_model_name}):")
            print(best_params)
            print("Mejor loss:", best_loss)
            
            print(f"\nPérdida normalizada del mejor modelo: {normalized_loss:.4f} (Valor máximo 0, mínimo 1)")
            print(f"Top 5 trials guardados en: {results_file}")

        elif choice == "5":
            print("¡Gracias por usar Pharmagen PModel!")
            sys.exit(0)

        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()