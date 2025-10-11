import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
import joblib
import sys
import json
import glob
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ======== AJUSTE DE PARÁMETROS ===========
# Puedes editar aquí los hiperparámetros principales:
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
EMB_DIM = 32
HIDDEN_DIM = 256
DATA_PATH = "var_pheno_ann_long.csv"
SAVE_MODEL_AS = "modelo_farmaco.pth"
SAVE_ENCODERS_AS = "encoders.pkl"
RESULTS_DIR = "../../results/"
# ==========================================

csv_files = glob.glob("../labels_vocabs/*.csv")

df = pd.concat([pd.read_csv(f, sep=';', usecols=['Drug', 'Genotype', 'Outcome', 'Variation'], index_col=False, dtype=str) for f in csv_files], ignore_index=True)


# ------------- PREDICTION INPUT -------------
# Puedes modificar esta lista para probar predicciones

predict_json = pd.read_json("drug_genes_random_list.json")

predict_json.columns = ['Drug', 'Genotype']

PREDICTION_INPUT = pd.DataFrame(predict_json)
# --------------------------------------------

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
    with open(str(Path('logs' + "training_log.txt")), "a", encoding="utf-8") as f: # type: ignore
        f.write(f"\nAverage loss: {avg_loss:.4f} | Effect accuracy: {acc_effect:.4f} | Outcome accuracy: {acc_outcome:.4f} | Avg accuracy: {avg_acc:.4f}")
        print((f"\nAverage loss: {avg_loss:.4f} | Effect accuracy: {acc_effect:.4f} | Outcome accuracy: {acc_outcome:.4f} | Avg accuracy: {avg_acc:.4f}"))

# Guarda los encoders y el modelo
joblib.dump(encoders, SAVE_ENCODERS_AS)
torch.save(model.state_dict(), SAVE_MODEL_AS)
print(f"\nModelo guardado como {SAVE_MODEL_AS} y encoders como {SAVE_ENCODERS_AS}")

# 5. PREDICCIÓN A PARTIR DE ENTRADAS DEFINIDAS
print("\n=== PREDICCIONES ===")
# Cargar modelo y encoders (para asegurar predicción tras entrenamiento)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoders = joblib.load(SAVE_ENCODERS_AS)
model = PharmacoModel(n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
model.load_state_dict(torch.load(SAVE_MODEL_AS, map_location=torch.device('cpu')))
model.eval()

# Normalizar y codificar entradas
input_df = pd.DataFrame(PREDICTION_INPUT, columns=['Drug', 'Genotype'], dtype=str, index=None)

# Normalizar genotipo usando equivalencias
input_df['Genotype'] = input_df['Genotype'].map(lambda x: equivalencias.get(x, x))
# Codificar Drug y Genotype
for col in ['Drug', 'Genotype']:
    if col in input_df:
        # Si el valor no está en los encoders, asigna -1 (desconocido)
        input_df[col] = input_df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
        

# Filtrar entradas válidas (que se hayan codificado correctamente)
valid_idx = (input_df['Drug'] != -1) & (input_df['Genotype'] != -1)
if not valid_idx.any():
    print("No hay entradas válidas para predecir (Drug o Genotype no reconocidos en el modelo).")
else:
    drug_tensor = torch.tensor(input_df.loc[valid_idx, 'Drug'].values, dtype=torch.long)
    genotype_tensor = torch.tensor(input_df.loc[valid_idx, 'Genotype'].values, dtype=torch.long)
    with torch.no_grad():
        effect_pred, outcome_pred = model(drug_tensor, genotype_tensor)
        effect_labels = torch.argmax(effect_pred, dim=1).numpy()
        outcome_labels = torch.argmax(outcome_pred, dim=1).numpy()
        # Decodificar los resultados
        effect_decoded = encoders['Effect'].inverse_transform(effect_labels)
        outcome_decoded = encoders['Outcome'].inverse_transform(outcome_labels)

    
    with open(str(Path(RESULTS_DIR + "predicciones.txt")), "w", encoding="utf-8") as f: # type: ignore
        for i in range(len(effect_decoded)):
            entrada_codificada = input_df[valid_idx].iloc[i].to_dict()
            # Decodifica Drug y Genotype usando los encoders inversos
            drug_name = encoders['Drug'].inverse_transform([entrada_codificada['Drug']])[0]
            genotype_name = encoders['Genotype'].inverse_transform([entrada_codificada['Genotype']])[0]
            result_str = (
            f"Input: {{'Drug': '{drug_name}', 'Genotype': '{genotype_name}'}}\n"
            f"  Predicted Effect:  {effect_decoded[i]}\n"
            f"  Predicted Outcome: {outcome_decoded[i]}\n"
            + "-" * 40 + "\n"
            )
            print(result_str, end="")  # Muestra también en consola
            f.write(result_str)

    print(f"\nPredicciones guardadas en {RESULTS_DIR}/predicciones.txt")