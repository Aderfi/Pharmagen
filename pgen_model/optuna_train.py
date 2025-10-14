"""-----------  FUNCION DE OPTUNA PARA EL ENTRENAMIENTO DEL MODELO  ----------------

        Esta función define el proceso de entrenamiento y evaluación del modelo
        utilizando los hiperparámetros sugeridos por Optuna.

------------------------------------------------------------------------------------"""

import torch
from torch.utils.data import DataLoader
from .data import PGenInputDataset, PGenDataset
from .model import PGenModel
from .train import train_model

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
                        n_drugs, 
                        n_genotypes, 
                        n_outcomes, 
                        n_variations, 
                        n_effects, 
                        n_entities,
                        params['emb_dim'], 
                        params['hidden_dim'], 
                        params['dropout_rate']
                        
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(

        train_loader, val_loader, model, optimizer, criterion,
        epochs=30, patience=5
    )
    
    return best_loss