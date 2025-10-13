import torch
from torch.utils.data import DataLoader
from pgen_model.data import PGenInputDataset, PGenDataset
from pgen_model.model import PGenModel
from pgen_model.train import train_model
from pgen_model.optuna_train import objective
import optuna
import pgen_model.metrics as metrics

def train_pipeline(PMODEL_DIR, csv_files, cols, params, epochs=30, patience=5, batch_size=8):
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
    print("""
    ================= Pharmagen PModel =================
    1. Entrenar modelo
    2. Realizar predicción (datos manuales)
    3. Realizar predicción (desde archivo)
    4. Salir
    ====================================================
    """)
    while True:
        choice = input("Selecciona opción (1-4): ").strip()
        if choice not in {"1","2","3","4"}:
            print("Opción no válida. Intente de nuevo.")
            continue

        if choice == "1":
            print("Iniciando entrenamiento...")
            # Aquí puedes pedir parámetros o usar los por defecto
            # Ejemplo mínimo (ajusta según tus necesidades reales)
            params = {
                'emb_dim': 64,
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'learning_rate': 3e-4
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
            for k, v in resultado.items():
                print(f"{k}: {v}")
        elif choice == "3":
            file_path = input("Ruta del archivo CSV: ")
            try:
                resultados = predict_from_file(file_path)
                print("\nResultados de las predicciones:")
                print(resultados)
            except Exception as e:
                print(f"Error procesando archivo: {e}")
        elif choice == "4":
            print("¡Gracias por usar Pharmagen PModel!")
            sys.exit(0)

if __name__ == "__main__":
    main()