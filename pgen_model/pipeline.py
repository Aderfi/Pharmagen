# coding=utf-8
"""------------------------------------------------------------------------------------
Pipeline de entrenamiento del modelo PharmacoGenómico. 
Utiliza las funciones definidas en data.py, model.py y train.py.
------------------------------------------------------------------------------------"""
import torch
from torch.utils.data import DataLoader
from .data import PGenInputDataset, PGenDataset, train_data_load
from .model import PGenModel
from .train import train_model
from .metrics import *


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
    
    
    output_dims = {col.lower(): df[col].nunique() for col in targets}
    
    model = PGenModel(
        n_drugs, n_genotypes, params['emb_dim'], params['hidden_dim'], params['dropout_rate'], output_dims
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss()
    best_loss = train_model(
        train_loader, val_loader, model, optimizer, criterion,
        epochs=epochs, patience=patience
    )
    print("Mejor loss en validación:", best_loss)
    with open(f'{RESULTS_DIR}/training_report.txt'.format(PMODEL_DIR), 'w') as f:
        f.write(f"Mejor loss en validación: {best_loss}\n")
        f.write(f"Hiperparámetros:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    return model