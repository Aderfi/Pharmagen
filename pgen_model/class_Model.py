import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import json
import optuna.trial, optuna.study

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

best_loss = float('inf')
trigger_times = 0

class PGenInputDataset():
    def __init__(self):
        self.drug = None
        self.genotype = None
        self.outcome = None
        self.variation = None
        self.effect = None
        self.entity = None
    
    def load_data(self, PMODEL_DIR, csv_files, cols, equivalencias):
        df = pd.concat([pd.read_csv(f, sep=';', usecols=cols, index_col=False, dtype=str) for f in csv_files], ignore_index=True)
        df["stratify_col"] = df["Effect"].astype(str) + "_" + df["Entity"].astype(str)
        print(f"Data loaded: {df.shape[0]} rows and {df.shape[1]} columns.")
        print(df.head(50))

        with open(f"{PMODEL_DIR}/geno_alleles_dict.json", "r") as f:
            equivalencias = json.load(f)

        df['Genotype'] = df['Genotype'].map(lambda x: equivalencias.get(x, x))

        # Encode Outcome, Variation, Effect, Entity (and Drug/Genotype) using LabelEncoder
        encoders = {}
        for col in df.columns:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col].astype(str))

        # Filter classes with very few examples according to Entity
        counts = df['stratify_col'].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df['stratify_col'].isin(suficientes)]

        print(f"Data after filtering classes with a single instance: {df.shape[0]} rows.")
        print(df.head())

        self.drug = torch.tensor(df['Drug'].values, dtype=torch.long)
        self.genotype = torch.tensor(df['Genotype'].values, dtype=torch.long)
        self.outcome = torch.tensor(df['Outcome'].values, dtype=torch.long)
        self.variation = torch.tensor(df['Variation'].values, dtype=torch.long)
        self.effect = torch.tensor(df['Effect'].values, dtype=torch.long)        
        self.entity = torch.tensor(df['Entity'].values, dtype=torch.long)


class PGenDataset(Dataset):
    def __init__(self, df):
        self.drug = torch.tensor(df['Drug'].values, dtype=torch.long)
        self.genotype = torch.tensor(df['Genotype'].values, dtype=torch.long)
        self.outcome = torch.tensor(df['Outcome'].values, dtype=torch.long)
        self.variation = torch.tensor(df['Variation'].values, dtype=torch.long)
        self.effect = torch.tensor(df['Effect'].values, dtype=torch.long)        
        self.entity = torch.tensor(df['Entity'].values, dtype=torch.long)

    def __len__(self):
        return len(self.drug)
    
    def __getitem__(self, idx):
        return {
            'drug': self.drug[idx],
            'genotype': self.genotype[idx],
            'outcome': self.outcome[idx],
            'variation': self.variation[idx],
            'effect': self.effect[idx],
            'entity': self.entity[idx],
        }

class PGenModel(nn.Module):
    def __init__(self, n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entitys,
                 emb_dim, hidden_dim, dropout_rate, device=device):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.outcome_out = nn.Linear(hidden_dim // 2, n_outcomes)
        self.variation_out = nn.Linear(hidden_dim // 2, n_variations)
        self.effect_out = nn.Linear(hidden_dim // 2, n_effects)
        self.entity_out = nn.Linear(hidden_dim // 2, n_entitys)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

    def forward_out_var(self, drug, genotype):
        drug_e = self.drug_emb(drug)
        geno_e = self.geno_emb(genotype)
        x = torch.cat([drug_e, geno_e], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        outcome = self.outcome_out(x)
        variation = self.variation_out(x)
        return outcome, variation
    
    def forward_eff_ent(self, drug, genotype):
        drug_e = self.drug_emb(drug)
        geno_e = self.geno_emb(genotype)
        x = torch.cat([drug_e, geno_e], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        effect = self.effect_out(x)
        entity = self.entity_out(x)
        return effect, entity


def iterations(train_loader, val_loader, model, optimizer, criterion, EPOCHS, PATIENCE):
    global best_loss, trigger_times
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_correct_outcome = 0
        total_correct_variation = 0
        total_correct_effect = 0
        total_correct_entity = 0
        total_samples = 0
        n_batches = len(train_loader)
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        for i, batch in enumerate(train_loader):
            drug = batch['drug'].to(device)
            genotype = batch['genotype'].to(device)
            outcome_pred = batch['outcome'].to(device)
            variation_pred = batch['variation'].to(device)
            effect = batch['effect'].to(device)
            entity = batch['entity'].to(device)

            optimizer.zero_grad()
            outcome_pred, variation_pred, effect_pred, entity_pred = model(drug, genotype)
            loss1 = criterion(outcome_pred, effect)
            loss2 = criterion(variation_pred, entity)
            loss3 = criterion(effect_pred, entity)
            loss4 = criterion(entity_pred, entity)
            loss = loss1 + loss2 + loss3 + loss4 
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                
            # Accuracy cálculo
            _, effect_pred_labels = torch.max(effect_pred, dim=1)
            _, entity_pred_labels = torch.max(entity_pred, dim=1)
            total_correct_effect += (effect_pred_labels == effect).sum().item()
            total_correct_entity += (entity_pred_labels == entity).sum().item()
            total_samples += effect.size(0)

            # Progreso con porcentaje
            percent = int(100 * (i + 1) / n_batches)
            bar = ('#' * (percent // 2)).ljust(50)
            sys.stdout.write(f"\r[{bar}] {percent}% - batch {i+1}/{n_batches} - Loss: {loss.item():.4f}")
            sys.stdout.flush()
        
        avg_loss = total_loss / n_batches
        acc_effect = total_correct_effect / total_samples
        acc_entity = total_correct_entity / total_samples
        avg_acc = (acc_effect + acc_entity) / 2

        # Evaluación en el conjunto de validación después de cada época
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:    
                drug = batch['drug'].to(device)
                genotype = batch['genotype'].to(device)
                effect = batch['effect'].to(device)
                entity = batch['entity'].to(device)
                effect_pred, entity_pred = model(drug, genotype)
                loss1 = criterion(effect_pred, effect)
                loss2 = criterion(entity_pred, entity)
                val_loss += (loss1 + loss2).item()
                val_samples += 1
        val_loss /= val_samples  # Promedio

        print(f" |  Validation loss: {val_loss:.4f}  |  Train loss: {avg_loss:.4f}  \
            Acc: {avg_acc:.4f} (E:{acc_effect:.4f}, En:{acc_entity:.4f})")
            
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print("Early stopping activado")
                break

    return best_loss