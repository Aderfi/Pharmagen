"""-----------  FUNCION DE OPTUNA PARA EL ENTRENAMIENTO DEL MODELO  ----------------

        Esta función define el proceso de entrenamiento y evaluación del modelo
        utilizando los hiperparámetros sugeridos por Optuna.

------------------------------------------------------------------------------------"""

import torch
from torch.utils.data import DataLoader
from .data import PGenInputDataset, PGenDataset, train_data_load
from .model import PGenModel
from .train import train_model

cols = ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']

def objective_all(trial):
    params = {
        'emb_dim': trial.suggest_categorical('emb_dim', [32, 64, 128]),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    targets = ['Outcome', 'Variation', 'Effect', 'Entity']
    
    csvfiles, df = train_data_load(targets)[:2]
    
    PMODEL_DIR = "./"
    #csv_files = ["train.csv"]
    #cols = ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']

    data_loader = PGenInputDataset()
    df = data_loader.load_data(PMODEL_DIR, csvfiles, cols)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_drugs = df['Drug'].nunique()
    n_genotypes = df['Genotype'].nunique()
    
    targets = [    cols = ['Drug', 'Genotype', 'Outcome', 'Variation']
'Outcome', 'Variation', 'Effect', 'Entity']
    output_dims = {col.lower(): df[col].nunique() for col in targets}

    model = PGenModel(
                        n_drugs, 
                        n_genotypes, 
                        params['emb_dim'], 
                        params['hidden_dim'], 
                        params['dropout_rate'],
                        output_dims
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(

        train_loader, val_loader, model, optimizer, criterion,
        epochs=30, patience=5
    )
    
    return best_loss

def objective_outcome_variation(trial):
    params = {
        'emb_dim': trial.suggest_categorical('emb_dim', [32, 64, 128]),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    cols = ['Drug', 'Genotype', 'Outcome', 'Variation']
    
    csvfiles, cols, df = train_data_load(cols)[:3]
    
    PMODEL_DIR = "./"
    #csv_files = ["train.csv"]

    data_loader = PGenInputDataset()
    df = data_loader.load_data(PMODEL_DIR, csvfiles, cols)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_drugs = df['Drug'].nunique()
    n_genotypes = df['Genotype'].nunique()

    targets = ['Outcome', 'Variation']
    output_dims = {col.lower(): df[col].nunique() for col in targets}
    
    model = PGenModel(
                        n_drugs, 
                        n_genotypes,  
                        params['emb_dim'], 
                        params['hidden_dim'], 
                        params['dropout_rate'],
                        output_dims
                        
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(

        train_loader, val_loader, model, optimizer, criterion,
        epochs=30, patience=5
    )
    
    return best_loss

def objective_effect_entity(trial):
    params = {
        'emb_dim': trial.suggest_categorical('emb_dim', [32, 64, 128]),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    csvfiles, cols, df = train_data_load()[:3]
    
    PMODEL_DIR = "./"
    #csv_files = ["train.csv"]
    cols = ['Drug', 'Genotype', 'Effect', 'Entity']

    data_loader = PGenInputDataset()
    df = data_loader.load_data(PMODEL_DIR, csvfiles, cols)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_drugs = df['Drug'].nunique()
    n_genotypes = df['Genotype'].nunique()
    n_effects = df['Effect'].nunique()
    n_entities = df['Entity'].nunique()

    targets = ['Effect', 'Entity']
    output_dims = {col.lower(): df[col].nunique() for col in targets}
    
    model = PGenModel(
                        n_drugs, 
                        n_genotypes, 
                        params['emb_dim'], 
                        params['hidden_dim'], 
                        params['dropout_rate'],
                        output_dims
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(

        train_loader, val_loader, model, optimizer, criterion,
        epochs=30, patience=5
    )
    
    return best_loss

def objective_variation_effect(trial):
    params = {
        'emb_dim': trial.suggest_categorical('emb_dim', [16, 32, 64, 128]),
        'hidden_dim': trial.suggest_int('hidden_dim', 64, 1024, step=64),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    }
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    csvfiles, cols, df = train_data_load()[:3]
    
    PMODEL_DIR = "./"
    #csv_files = ["train.csv"]
    cols = ['Drug', 'Genotype', 'Variation', 'Effect']

    data_loader = PGenInputDataset()
    df = data_loader.load_data(PMODEL_DIR, csvfiles, cols)
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_dataset = PGenDataset(train_df)
    val_dataset = PGenDataset(val_df)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    n_drugs = df['Drug'].nunique()
    n_genotypes = df['Genotype'].nunique()
    n_variations = df['Variation'].nunique()
    n_effects = df['Effect'].nunique()

    targets = ['Variation', 'Effect']
    output_dims = {col.lower(): df[col].nunique() for col in targets}

    model = PGenModel(
                        n_drugs, 
                        n_genotypes, 
                        params['emb_dim'], 
                        params['hidden_dim'], 
                        params['dropout_rate'],
                        output_dims
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(
        train_loader, val_loader, model, optimizer, criterion,
        epochs=50, patience=10
    )
    
    return best_loss