import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import sys
import json
import glob
from pathlib import Path
import optuna.trial
from optuna import Trial, Study

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# ======== AJUSTE DE PARÁMETROS ===========
# Puedes editar aquí los hiperparámetros principales:
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1.5258586638387263e-05
EMB_DIM = 64
HIDDEN_DIM = 704
DROPOUT_RATE = 0.49086195278699424
PATIENCE = 10

# EMB_DIM = OPORTUNA: 64
# HIDDEN_DIM = OPORTUNA: 704
# LEARNING_RATE = OPORTUNA: 2.996788185777191e-05
# DROPOUT_RATE = OPORTUNA: 0.2893325389845274

DATA_PATH = "var_pheno_ann_long.csv"
SAVE_MODEL_AS = "modelo_effect_entity.pth"
SAVE_ENCODERS_AS = "encoders_effect_entity.pkl"
RESULTS_DIR = "../../results/"
# ==========================================

csv_files = glob.glob("../labels_vocabs/*.csv")

df = pd.concat(
    [
        pd.read_csv(
            f,
            sep=";",
            usecols=["Drug", "Genotype", "Effect", "Entity"],
            index_col=False,
            dtype=str,
        )
        for f in csv_files
    ],
    ignore_index=True,
)


# ------------- PREDICTION INPUT -------------
# Puedes modificar esta lista para probar predicciones

predict_json = pd.read_json("drug_genes_random_list.json")

predict_json.columns = ["Drug", "Genotype"]

PREDICTION_INPUT = pd.DataFrame(predict_json)
# --------------------------------------------

# Cargar equivalencias de genotipos
with open("geno_alleles_dict.json", "r") as f:
    equivalencias = json.load(f)

# 1. Cargar y preprocesar los datos
df = pd.concat(
    [
        pd.read_csv(
            f,
            sep=";",
            usecols=["Drug", "Genotype", "Effect", "Entity"],
            index_col=False,
            dtype=str,
        )
        for f in csv_files
    ],
    ignore_index=True,
)
df["stratify_col"] = df["Effect"] + "_" + df["Entity"]


df["Genotype"] = df["Genotype"].map(lambda x: equivalencias.get(x, x))

# Codifica las variables categóricas a índices numéricos
encoders = {}
for col in df.columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col].astype(str))


counts = df["Entity"].value_counts()
suficientes = counts[counts > 1].index
df = df[df["Entity"].isin(suficientes)]


df_train, df_val = train_test_split(df, test_size=0.2, stratify=df["stratify_col"])


# 2. Definir Dataset personalizado
class PharmacoDataset(Dataset):
    def __init__(self, df):
        self.drug = torch.tensor(df["Drug"].values, dtype=torch.long)
        self.genotype = torch.tensor(df["Genotype"].values, dtype=torch.long)
        self.effect = torch.tensor(df["Effect"].values, dtype=torch.long)
        self.entity = torch.tensor(df["Entity"].values, dtype=torch.long)

    def __len__(self):
        return len(self.drug)

    def __getitem__(self, idx):
        return {
            "drug": self.drug[idx],
            "genotype": self.genotype[idx],
            "effect": self.effect[idx],
            "entity": self.entity[idx],
        }


dataset = PharmacoDataset(df)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

train_dataset = PharmacoDataset(df_train)
val_dataset = PharmacoDataset(df_val)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 3. Definir el modelo
class PharmacoModel(nn.Module):
    def __init__(
        self, n_drugs, n_genotypes, n_effects, n_entitys, emb_dim=8, hidden_dim=64
    ):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.effect_out = nn.Linear(hidden_dim // 2, n_effects)
        self.entity_out = nn.Linear(hidden_dim // 2, n_entitys)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, drug, genotype):
        drug_e = self.drug_emb(drug)
        geno_e = self.geno_emb(genotype)
        x = torch.cat([drug_e, geno_e], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        effect = self.effect_out(x)
        entity = self.entity_out(x)
        return effect, entity


n_drugs = len(encoders["Drug"].classes_)
n_genotypes = len(encoders["Genotype"].classes_)
n_effects = len(encoders["Effect"].classes_)
n_entitys = len(encoders["Entity"].classes_)


"""
import optuna
emb_dim = trial.suggest_categorical('emb_dim', [8, 16, 32, 64])
hidden_dim = trial.suggest_int('hidden_dim', 64, 1024, step=64)
lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
# ...arma y entrena tu modelo usando estos valores...
# ...devuelve la valid_loss o 1 - valid_accuracy...
"""

model = PharmacoModel(
    n_drugs, n_genotypes, n_effects, n_entitys, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

# 4. Entrenamiento con barra de progreso y verbose, mostrando average loss y accuracy

best_loss = float("inf")  # Puedes ajustar cuántas épocas esperar sin mejora
trigger_times = 0

for epoch in range(EPOCHS):

    """
    Entrenamiento del modelo con barra de progreso y verbose
    Muestra average loss, accuracy por tarea y accuracy promedio
    Guarda los resultados en logs/training_log.txt
    """

    model.train()
    total_loss = 0
    total_correct_effect = 0
    total_correct_entity = 0
    total_samples = 0
    n_batches = len(loader)
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    for i, batch in enumerate(loader):
        drug = batch["drug"]
        genotype = batch["genotype"]
        effect = batch["effect"]
        entity = batch["entity"]

        optimizer.zero_grad()
        effect_pred, entity_pred = model(drug, genotype)
        loss1 = criterion(effect_pred, effect)
        loss2 = criterion(entity_pred, entity)
        loss = loss1 + loss2
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
        bar = ("#" * (percent // 2)).ljust(50)
        sys.stdout.write(
            f"\r[{bar}] {percent}% - batch {i+1}/{n_batches} - Loss: {loss.item():.4f}"
        )
        sys.stdout.flush()
    avg_loss = total_loss / n_batches
    acc_effect = total_correct_effect / total_samples
    acc_entity = total_correct_entity / total_samples
    avg_acc = (acc_effect + acc_entity) / 2

    """
    Evaluación en el conjunto de validación
    Muestra la pérdida de validación al final de cada época
    """

    model.eval()
    val_loss = 0
    val_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            drug = batch["drug"]
            genotype = batch["genotype"]
            effect = batch["effect"]
            entity = batch["entity"]
            effect_pred, entity_pred = model(drug, genotype)
            loss1 = criterion(effect_pred, effect)
            loss2 = criterion(entity_pred, entity)
            val_loss += (loss1 + loss2).item()
            val_samples += 1
    val_loss /= val_samples  # Promedio

    print(f" |  Validation loss: {val_loss:.4f}")

    if val_loss < best_loss:
        best_loss = val_loss
        trigger_times = 0
        # torch.save(model.state_dict(), "best_model.pth")  # Opcional: guarda el mejor modelo
    else:
        trigger_times += 1
        if trigger_times >= PATIENCE:
            print("Early stopping activado")
            break

    with open(str(Path("logs" + "training_log.txt")), "a", encoding="utf-8") as f:  # type: ignore
        f.write(
            f"\nAverage loss: {avg_loss:.4f} | Effect accuracy: {acc_effect:.4f} | Entity accuracy: {acc_entity:.4f} | Avg accuracy: {avg_acc:.4f}"
        )
        print(
            (
                f"\nAverage loss: {avg_loss:.4f} | Effect accuracy: {acc_effect:.4f} | Entity accuracy: {acc_entity:.4f} | Avg accuracy: {avg_acc:.4f}"
            )
        )

"""TRIAL PARA MEJORES HIPERPARÁMETROS"""
"""


def objective(trial):
    emb_dim = trial.suggest_categorical('emb_dim', [8, 16, 32, 64])
    hidden_dim = trial.suggest_int('hidden_dim', 64, 1024, step=64)
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

    model = PharmacoModel(n_drugs, n_genotypes, n_effects, n_entitys, emb_dim=emb_dim, hidden_dim=hidden_dim)
    model.dropout = nn.Dropout(dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float('inf')
    trigger_times = 0

    for epoch in range(EPOCHS):
        model.train()
        for batch in train_loader:
            drug = batch['drug']
            genotype = batch['genotype']
            effect = batch['effect']
            entity = batch['entity']

            optimizer.zero_grad()
            effect_pred, entity_pred = model(drug, genotype)
            loss1 = criterion(effect_pred, effect)
            loss2 = criterion(entity_pred, entity)
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()

        # Validación
        model.eval()
        val_loss = 0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                drug = batch['drug']
                genotype = batch['genotype']
                effect = batch['effect']
                entity = batch['entity']
                effect_pred, entity_pred = model(drug, genotype)
                loss1 = criterion(effect_pred, effect)
                loss2 = criterion(entity_pred, entity)
                val_loss += (loss1 + loss2).item()
                val_samples += 1
        val_loss /= max(val_samples, 1)

        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= PATIENCE:
                break

    return best_loss
"""
# Guarda los encoders y el modelo
joblib.dump(encoders, SAVE_ENCODERS_AS)
torch.save(model.state_dict(), SAVE_MODEL_AS)
print(f"\nModelo guardado como {SAVE_MODEL_AS} y encoders como {SAVE_ENCODERS_AS}")

# 5. PREDICCIÓN A PARTIR DE ENTRADAS DEFINIDAS
print("\n=== PREDICCIONES ===")
# Cargar modelo y encoders (para asegurar predicción tras entrenamiento)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoders = joblib.load(SAVE_ENCODERS_AS)
model = PharmacoModel(
    n_drugs, n_genotypes, n_effects, n_entitys, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM
)
model.load_state_dict(torch.load(SAVE_MODEL_AS, map_location=torch.device("cpu")))
model.eval()

# Normalizar y codificar entradas
input_df = pd.DataFrame(
    PREDICTION_INPUT, columns=["Drug", "Genotype"], dtype=str, index=None
)

# Normalizar genotipo usando equivalencias
input_df["Genotype"] = input_df["Genotype"].map(lambda x: equivalencias.get(x, x))
# Codificar Drug y Genotype
for col in ["Drug", "Genotype"]:
    if col in input_df:
        # Si el valor no está en los encoders, asigna -1 (desconocido)
        input_df[col] = input_df[col].map(
            lambda x: (
                encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
            )
        )


# Filtrar entradas válidas (que se hayan codificado correctamente)
valid_idx = (input_df["Drug"] != -1) & (input_df["Genotype"] != -1)
if not valid_idx.any():
    print(
        "No hay entradas válidas para predecir (Drug o Genotype no reconocidos en el modelo)."
    )
else:
    drug_tensor = torch.tensor(input_df.loc[valid_idx, "Drug"].values, dtype=torch.long)
    genotype_tensor = torch.tensor(
        input_df.loc[valid_idx, "Genotype"].values, dtype=torch.long
    )
    with torch.no_grad():
        effect_pred, entity_pred = model(drug_tensor, genotype_tensor)
        effect_labels = torch.argmax(effect_pred, dim=1).numpy()
        entity_labels = torch.argmax(entity_pred, dim=1).numpy()
        # Decodificar los resultados
        effect_decoded = encoders["Effect"].inverse_transform(effect_labels)
        entity_decoded = encoders["Entity"].inverse_transform(entity_labels)

    with open(str(Path(RESULTS_DIR + "predicciones.txt")), "w", encoding="utf-8") as f:  # type: ignore
        for i in range(len(effect_decoded)):
            entrada_codificada = input_df[valid_idx].iloc[i].to_dict()
            # Decodifica Drug y Genotype usando los encoders inversos
            drug_name = encoders["Drug"].inverse_transform(
                [entrada_codificada["Drug"]]
            )[0]
            genotype_name = encoders["Genotype"].inverse_transform(
                [entrada_codificada["Genotype"]]
            )[0]
            result_str = (
                f"Input: {{'Drug': '{drug_name}', 'Genotype': '{genotype_name}'}}\n"
                f"  Predicted Effect:  {effect_decoded[i]}\n"
                f"  Predicted Entity: {entity_decoded[i]}\n" + "-" * 40 + "\n"
            )
            print(result_str, end="")  # Muestra también en consola
            f.write(result_str)

    print(f"\nPredicciones guardadas en {RESULTS_DIR}/predicciones.txt")
"""
# 6. OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print("Mejores hiperparámetros:", study.best_params)
"""
