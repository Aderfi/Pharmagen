"""
Punto de entrada para el paquete pgen_model.
Proporciona un menú CLI para entrenar y utilizar el modelo predictivo.
"""

import sys
import torch
from torch.utils.data import DataLoader

from .data import PGenInputDataset, PGenDataset
from .model import PGenModel
from .train import train_model
from .optuna_train import objective_all, objective_outcome_variation, objective_effect_entity, objective_variation_effect
from .predict import predict_single_input, predict_from_file   # Asegúrate de implementar estas funciones
import optuna
from .metrics import metrics_models
from .pipeline import train_pipeline

def main():
    model_choice = ""
    model_name = ""

    print("""
        ================= Pharmagen PModel =================
        SELECCIONA EL MODELO QUE QUIERES UTILIZAR:
        1-- Outcome-Variation-Effect-Entity
        2-- Outcome-Variation
        3-- Effect-Entity
        4-- Variation-Effect
        ====================================================
        """)
    model_choice = ""
    while model_choice not in ["1", "2", "3", "4"]:
            model_choice = str(input("Selecciona opción (1-4): ").strip())
            if model_choice == "1":
                model_name = "Effect-Entity-Outcome-Variation"
            elif model_choice == "2":
                model_name = "Outcome-Variation"
            elif model_choice == "3":
                model_name = "Effect-Entity"
            elif model_choice == "4":
                model_name = "Variation-Effect"
            else:
                print("Opción no válida. Intente de nuevo.")
            
    
    while True:
        print(f"""
        ================= Pharmagen PModel =================
        1. Entrenar modelo   ---->     {model_name}
        2. Realizar predicción (datos manuales)
        3. Realizar predicción (desde archivo)
        4. Optimizacion de hiperparámetros (Optuna)
        5. Salir
        ====================================================
        """)
                
        choice = input("Selecciona opción (1-5): ").strip()
        if choice == "1":
            print("Iniciando entrenamiento...")
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
            PMODEL_DIR = "./"
            csv_files = ["train.csv"]
            cols = ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']
            train_pipeline(PMODEL_DIR, csv_files, cols, params)
            
        ##########################################################
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
        ##########################################################
        elif choice == "3":
            file_path = input("Ruta del archivo CSV: ")
            try:
                resultados = predict_from_file(file_path)
                print("\nResultados de las predicciones:")
                print(resultados)
            except Exception as e:
                print(f"Error procesando archivo: {e}")
        #########################################################        
        elif choice == "4":
            print("Optimizando hiperparámetros con Optuna...")
            
            print("""
            ================= Pharmagen PModel =================
            SELECCIONA EL MODELO QUE QUIERES UTILIZAR:
            1-- Outcome-Variation-Effect-Entity
            2-- Outcome-Variation
            3-- Effect-Entity
            4-- Variation-Effect
            ====================================================
            """)
            
            while model_choice not in ["1", "2", "3", "4"]:
                    wmodel_choice = input("Selecciona el modelo para optimización (1-4): ").strip()

            objective = objective_all if model_choice == "1" else \
                        objective_outcome_variation if model_choice == "2" else \
                        objective_effect_entity if model_choice == "3" else \
                        objective_variation_effect if model_choice == "4" else None
                        

            study = optuna.create_study(direction="minimize")
            study.optimize(objective_all, n_trials=100)
            print("Mejores hiperparámetros encontrados:")
            print(study.best_trial.params)
            print("Mejor loss:", study.best_trial.value)
        elif choice == "5":
            print("¡Gracias por usar Pharmagen PModel!")
            sys.exit(0)
        else:
            print("Opción no válida. Intente de nuevo.")
        ###########################################################
if __name__ == "__main__":
    main()