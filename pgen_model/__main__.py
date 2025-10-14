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
from .optuna_train import objective
from .predict import predict_single_input, predict_from_file   # Asegúrate de implementar estas funciones
import optuna
import pgen_model.metrics as metrics    
from pgen_model.metrics import metrics_models

def train_pipeline(PMODEL_DIR, csv_files, cols, params, epochs=30, patience=5, batch_size=8):
    """
    Pipeline de entrenamiento: carga datos, inicializa modelo, entrena y guarda resultados.
    """
    data_loader = PGenInputDataset()
    df = data_loader.load_data(PMODEL_DIR, csv_files, cols)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    n_drugs = df['Drug'].nunique()
    n_genotypes = df['Genotype'].nunique()
    n_outcomes = df['Outcome'].nunique()
    n_variations = df['Variation'].nunique()
    n_effects = df['Effect'].nunique()
    n_entities = df['Entity'].nunique()
    model = PGenModel(
        n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entities,
        params['emb_dim'], params['hidden_dim'], params['dropout_rate']
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(
        train_loader, val_loader, model, optimizer, criterion,
        epochs=epochs, patience=patience
    )
    print("Mejor loss en validación:", best_loss)
    with open(f'{metrics.RESULTS_DIR}/training_report.txt'.format(PMODEL_DIR), 'w') as f:
        f.write(f"Mejor loss en validación: {best_loss}\n")
        f.write(f"Hiperparámetros:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    return model

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
            print("Optimizando hiperparámetros con Optuna...")
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20)
            print("Mejores hiperparámetros encontrados:")
            print(study.best_trial.params)
            print("Mejor loss:", study.best_trial.value)
        elif choice == "5":
            print("¡Gracias por usar Pharmagen PModel!")
            sys.exit(0)
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()