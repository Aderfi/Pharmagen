import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import json
import optuna
from optuna import Trial as trial
from optuna import Study as study

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ======== AJUSTE DE PARÁMETROS ===========
EPOCHS = 100
PATIENCE = 10

Dat = "var_pheno_ann_long.csv"
SAVE_MODEL_AS = "modelo_farmaco.pth"
SAVE_ENCODERS_AS = "encoders_.pkl"
RESULTS_DIR = "../../results/"
# ==========================================

csv_files = ["var_pheno_ann_model.csv", "var_drug_ann_model.csv"]
print(csv_files)

# Carga solo las columnas de trabajo en el orden deseado
cols = ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']
df = pd.concat([pd.read_csv(f, sep=';', usecols=cols, index_col=False, dtype=str) for f in csv_files], ignore_index=True)

df["stratify_col"] = df["Variation"].astype(str) + "_" + df["Effect"].astype(str)

print(f"Datos cargados: {df.shape[0]} filas y {df.shape[1]} columnas.")
print(df.head(50))

predict_json = pd.read_json("drug_genes_random_list.json")
predict_json.columns = ['Drug', 'Genotype']
PREDICTION_INPUT = pd.DataFrame(predict_json)

with open("geno_alleles_dict.json", "r") as f:
    equivalencias = json.load(f)

df['Genotype'] = df['Genotype'].map(lambda x: equivalencias.get(x, x))

# Codifica Outcome, Variation, Effect, Entity (y Drug/Genotype) usando LabelEncoder
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# Filtra clases con muy pocos ejemplos según Entity
counts = df['stratify_col'].value_counts()
suficientes = counts[counts > 1].index
df = df[df['stratify_col'].isin(suficientes)]

print(f"Datos tras filtrar clases con una sola instancia: {df.shape[0]} filas.")
print(df.head())

# Chequea que los targets estén en el rango correcto
'''
for col, n in zip(['Outcome', 'Variation', 'Effect', 'Entity'],
                  [len(encoders['Outcome'].classes_), len(encoders['Variation'].classes_),
                   len(encoders['Effect'].classes_), len(encoders['Entity'].classes_)]):
    print(f"{col}: min={df[col].min()}, max={df[col].max()}, n_classes={n}")
    assert df[col].min() >= 0
    assert df[col].max() < n
'''

df_train, df_val = train_test_split(df, test_size=0.2, stratify=df['stratify_col'], random_state=42)

class PharmacoDataset(Dataset):
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
            'entity': self.entity[idx]
        }

n_drugs = len(encoders['Drug'].classes_)
n_genotypes = len(encoders['Genotype'].classes_)
n_outcomes = len(encoders['Outcome'].classes_)
n_variations = len(encoders['Variation'].classes_)
n_effects = len(encoders['Effect'].classes_)
n_entitys = len(encoders['Entity'].classes_)

class PharmacoModel(nn.Module):
    def __init__(self, n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entitys,
                 emb_dim=8, hidden_dim=64, dropout_rate=0.3, device=device):
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

    def forward(self, drug, genotype):
        drug_e = self.drug_emb(drug)
        geno_e = self.geno_emb(genotype)
        x = torch.cat([drug_e, geno_e], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        outcome = self.outcome_out(x)
        variation = self.variation_out(x)
        effect = self.effect_out(x)
        entity = self.entity_out(x)
        return outcome, variation, effect, entity

def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [32,64,128,256])
    emb_dim = trial.suggest_categorical('emb_dim', [8, 16, 32, 64])
    hidden_dim = trial.suggest_int('hidden_dim', 64, 1024, step=64)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    # DataLoaders con el batch_size elegido por Optuna
    train_dataset = PharmacoDataset(df_train)
    val_dataset = PharmacoDataset(df_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PharmacoModel(n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entitys,
                          emb_dim=emb_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float('inf')
    trigger_times = 0

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
            outcome = batch['outcome'].to(device)
            variation = batch['variation'].to(device)
            effect = batch['effect'].to(device)
            entity = batch['entity'].to(device)

            optimizer.zero_grad()
            outcome_pred, variation_pred, effect_pred, entity_pred = model(drug, genotype)
            loss1 = criterion(outcome_pred, outcome)
            loss2 = criterion(variation_pred, variation)
            loss3 = criterion(effect_pred, effect)
            loss4 = criterion(entity_pred, entity)
            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
                
            # Accuracy cálculo
            _, outcome_pred_labels = torch.max(outcome_pred, dim=1)
            _, variation_pred_labels = torch.max(variation_pred, dim=1)
            _, effect_pred_labels = torch.max(effect_pred, dim=1)
            _, entity_pred_labels = torch.max(entity_pred, dim=1)
            total_correct_outcome += (outcome_pred_labels == outcome).sum().item()
            total_correct_variation += (variation_pred_labels == variation).sum().item()
            total_correct_effect += (effect_pred_labels == effect).sum().item()
            total_correct_entity += (entity_pred_labels == entity).sum().item()
            total_samples += outcome.size(0)
                
            # Progreso con porcentaje
            percent = int(100 * (i + 1) / n_batches)
            bar = ('#' * (percent // 2)).ljust(50)
            sys.stdout.write(f"\r[{bar}] {percent}% - batch {i+1}/{n_batches} - Loss: {loss.item():.4f}")
            sys.stdout.flush()
        
        avg_loss = total_loss / n_batches
        acc_outcome = total_correct_outcome / total_samples
        acc_variation = total_correct_variation / total_samples
        acc_effect = total_correct_effect / total_samples
        acc_entity = total_correct_entity / total_samples
        avg_acc = (acc_outcome + acc_variation + acc_effect + acc_entity) / 4

        # Evaluación en el conjunto de validación después de cada época
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:    
                drug = batch['drug'].to(device)
                genotype = batch['genotype'].to(device)
                outcome = batch['outcome'].to(device)
                variation = batch['variation'].to(device)
                effect = batch['effect'].to(device)
                entity = batch['entity'].to(device)
                outcome_pred, variation_pred, effect_pred, entity_pred = model(drug, genotype)
                loss1 = criterion(outcome_pred, outcome)
                loss2 = criterion(variation_pred, variation)
                loss3 = criterion(effect_pred, effect)
                loss4 = criterion(entity_pred, entity)
                val_loss += (loss2 + loss3).item()
                val_samples += 1
        val_loss /= val_samples  # Promedio

        print(f" |  Validation loss: {val_loss:.4f}  |  Train loss: {avg_loss:.4f}  \
            Acc: {avg_acc:.4f} (O:{acc_outcome:.4f}, V:{acc_variation:.4f}, E:{acc_effect:.4f}, En:{acc_entity:.4f})")
            
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                print("Early stopping activado")
                break

    return best_loss

# 6. OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
print("Mejores hiperparámetros:", study.best_params)

with open("optuna_results.txt", "a") as f:
    f.write("\n---- Trial Modelo Outcome-Variation-Effect-Entity-----\n")
    f.write(f"Mejores hiperparámetros: {study.best_params}\n")