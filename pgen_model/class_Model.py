import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import json
from src.config.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_loss = float('inf')
trigger_times = 0

class PGenInputDataset:
    """
    Carga y codifica los datos de entrada.
    """
    def __init__(self):
        self.data = None
        self.encoders = {}

    def load_data(self, PMODEL_DIR, csv_files, cols):
        
        df = pd.concat([pd.read_csv(f, sep=';', usecols=cols, index_col=False, dtype=str) for f in csv_files], ignore_index=True)
        df_cache = df.copy()
        df_cache.pop('Drug')
        df_cache.pop('Genotype')
        df["stratify_col"] = df_cache.apply(lambda x: "_".join(x.astype(str)), axis=1)

        with open(f"{PMODEL_DIR}/train_data/json_dicts/geno_alleles_dict.json", "r") as f:
            equivalencias = json.load(f)
            
        df['Genotype'] = df['Genotype'].map(lambda x: equivalencias.get(x, x))

        # Codifica todas las columnas con LabelEncoder
        for col in df.columns:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))

        # Filtra clases con una sola instancia
        counts = df['stratify_col'].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df['stratify_col'].isin(suficientes)]

        # self.data = df.reset_index(drop=True)
        self.data = df.drop(columns=['stratify_col']).reset_index(drop=True)
        
        return self.data

    def get_tensors(self):
        tensors = {}
        for col in ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']:
            tensors[col.lower()] = torch.tensor(self.data[col].values, dtype=torch.long)
            
        return tensors

class PGenDataset(Dataset):
    """
    Dataset PyTorch para el modelo.
    """
    def __init__(self, df):
        self.tensors = {col.lower(): torch.tensor
                        (df[col].values, dtype=torch.long)
                        for col in 
                        ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']}

    def __len__(self):
        return len(self.tensors['drug'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}

class PGenModel(nn.Module):
    """
    Red neuronal para predicción de Outcome, Variation, Effect y Entity.
    """
    def __init__(self, n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entities, emb_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.outcome_out = nn.Linear(hidden_dim // 2, n_outcomes) if n_outcomes.exists() else None
        self.variation_out = nn.Linear(hidden_dim // 2, n_variations) if n_variations else None
        self.effect_out = nn.Linear(hidden_dim // 2, n_effects) if n_effects else None
        self.entity_out = nn.Linear(hidden_dim // 2, n_entities) if n_entities else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, drug, genotype):
        x = torch.cat([self.drug_emb(drug), self.geno_emb(genotype)], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return {
            'outcome': self.outcome_out(x) if self.outcome_out else None,
            'variation': self.variation_out(x) if self.variation_out else None,
            'effect': self.effect_out(x) if self.effect_out else None,
            'entity': self.entity_out(x) if self.entity_out else None
        }

def train_model(train_loader, val_loader, model, optimizer, criterion, epochs, patience, device=device):
    """
    Entrena el modelo con early stopping.
    """
    best_loss = float('inf')
    trigger_times = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            drug = batch['drug'].to(device)
            genotype = batch['genotype'].to(device)
            targets = {k: batch[k].to(device) for k in ['outcome', 'variation', 'effect', 'entity']}
            optimizer.zero_grad()
            outputs = model(drug, genotype)
            loss = sum(criterion(outputs[k], targets[k]) for k in outputs)
            loss.backward() #type: ignore
            optimizer.step()
            total_loss += loss.item() # type: ignore
        avg_loss = total_loss / len(train_loader)
        
        #------------------------------------#
        #------- Validación del modelo ------#
        #------------------------------------#
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug'].to(device)
                genotype = batch['genotype'].to(device)
                targets = {k: batch[k].to(device) for k in ['effect', 'entity']}
                outputs = model(drug, genotype)
                loss = sum(criterion(outputs[k], targets[k]) for k in ['effect', 'entity'])
                val_loss += loss.item() # type: ignore
        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping activado.")
                break
    return best_loss