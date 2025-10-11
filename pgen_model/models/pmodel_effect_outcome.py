import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import json

# ======== AJUSTE DE PARÁMETROS ===========
# Puedes editar aquí los hiperparámetros principales:
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
EMB_DIM = 16
HIDDEN_DIM = 128
DATA_PATH = "var_pheno_ann_long.csv"
SAVE_MODEL_AS = "modelo_farmaco.pth"
SAVE_ENCODERS_AS = "encoders.pkl"
# ==========================================
# Cargar equivalencias de genotipos
with open("geno_alleles_dict.json", "r") as f:
    equivalencias = json.load(f)

# 1. Cargar y preprocesar los datos
df = pd.read_csv(DATA_PATH, sep=';')
df = df[['Drug', 'Genotype', 'Effect', 'Outcome']]  # Selecciona solo las columnas relevantes

df['Genotype'] = df['Genotype'].map(lambda x: equivalencias.get(x, x))

# Codifica las variables categóricas a índices numéricos
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))

# 2. Definir Dataset personalizado
class PharmacoDataset(Dataset):
    def __init__(self, df):
        self.drug = torch.tensor(df['Drug'].values, dtype=torch.long)
        self.genotype = torch.tensor(df['Genotype'].values, dtype=torch.long)
        self.effect = torch.tensor(df['Effect'].values, dtype=torch.long)
        self.outcome = torch.tensor(df['Outcome'].values, dtype=torch.long)
    
    def __len__(self):
        return len(self.drug)
    
    def __getitem__(self, idx):
        return {
            'drug': self.drug[idx],
            'genotype': self.genotype[idx],
            'effect': self.effect[idx],
            'outcome': self.outcome[idx]
        }

dataset = PharmacoDataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# 3. Definir el modelo
class PharmacoModel(nn.Module):
    def __init__(self, n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.effect_out = nn.Linear(hidden_dim//2, n_effects)
        self.outcome_out = nn.Linear(hidden_dim//2, n_outcomes)
        self.relu = nn.ReLU()
    
    def forward(self, drug, genotype):
        drug_e = self.drug_emb(drug)
        geno_e = self.geno_emb(genotype)
        x = torch.cat([drug_e, geno_e], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        effect = self.effect_out(x)
        outcome = self.outcome_out(x)
        return effect, outcome

n_drugs = len(encoders['Drug'].classes_)
n_genotypes = len(encoders['Genotype'].classes_)
n_effects = len(encoders['Effect'].classes_)
n_outcomes = len(encoders['Outcome'].classes_)

model = PharmacoModel(n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. Entrenamiento con barra de progreso y verbose
# 4. Entrenamiento con barra de progreso y verbose, mostrando average loss y accuracy
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_correct_effect = 0
    total_correct_outcome = 0
    total_samples = 0
    n_batches = len(loader)
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    for i, batch in enumerate(loader):
        drug = batch['drug']
        genotype = batch['genotype']
        effect = batch['effect']
        outcome = batch['outcome']
        
        optimizer.zero_grad()
        effect_pred, outcome_pred = model(drug, genotype)
        loss1 = criterion(effect_pred, effect)
        loss2 = criterion(outcome_pred, outcome)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Accuracy cálculo
        _, effect_pred_labels = torch.max(effect_pred, dim=1)
        _, outcome_pred_labels = torch.max(outcome_pred, dim=1)
        total_correct_effect += (effect_pred_labels == effect).sum().item()
        total_correct_outcome += (outcome_pred_labels == outcome).sum().item()
        total_samples += effect.size(0)
        
        # Progreso con porcentaje
        percent = int(100 * (i + 1) / n_batches)
        bar = ('#' * (percent // 2)).ljust(50)
        sys.stdout.write(f"\r[{bar}] {percent}% - batch {i+1}/{n_batches} - Loss: {loss.item():.4f}")
        sys.stdout.flush()
    avg_loss = total_loss / n_batches
    acc_effect = total_correct_effect / total_samples
    acc_outcome = total_correct_outcome / total_samples
    avg_acc = (acc_effect + acc_outcome) / 2
    print(f"\nAverage loss: {avg_loss:.4f} | Effect accuracy: {acc_effect:.4f} | Outcome accuracy: {acc_outcome:.4f} | Avg accuracy: {avg_acc:.4f}")

# Guarda los encoders y el modelo
joblib.dump(encoders, SAVE_ENCODERS_AS)
torch.save(model.state_dict(), SAVE_MODEL_AS)
print(f"\nModelo guardado como {SAVE_MODEL_AS} y encoders como {SAVE_ENCODERS_AS}")