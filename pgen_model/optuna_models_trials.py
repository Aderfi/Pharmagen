import torch
from torch.utils.data import DataLoader
from p_genmodel import PGenInputDataset, PGenDataset, PGenModel, train_model
import optuna

def train_pipeline(PMODEL_DIR, csv_files, cols, params, epochs=30, patience=5, batch_size=8):
    # 1. Carga y procesado de datos
    data_loader = PGenInputDataset()
    df = data_loader.load_data(PMODEL_DIR, csv_files, cols)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # 2. Inicialización del modelo
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

    # 3. Entrenamiento
    best_loss = train_model(
        train_loader, val_loader, model, optimizer, criterion,
        epochs=epochs, patience=patience
    )
    print("Mejor loss en validación:", best_loss)
    return model

# ----- Optimización con Optuna -----
def objective(trial):
    params = {
        'emb_dim': trial.suggest_categorical('emb_dim', [32, 64, 128]),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    PMODEL_DIR = "./"
    csv_files = ["train.csv"]
    cols = ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']

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
        epochs=30, patience=5
    )
    return best_loss

if __name__ == "__main__":
    # Entrenamiento normal
    params = {
        'emb_dim': 64,
        'hidden_dim': 256,
        'dropout_rate': 0.3,
        'learning_rate': 3e-4
    }
    PMODEL_DIR = "./"
    csv_files = ["train.csv"]
    cols = ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']

    print("=== Entrenando con hiperparámetros fijos ===")
    model = train_pipeline(PMODEL_DIR, csv_files, cols, params)

    # Entrenamiento con Optuna
    print("\n=== Optimizando con Optuna ===")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    print("Mejores hiperparámetros encontrados:")
    print(study.best_trial.params)
    print("Mejor loss:", study.best_trial.value)