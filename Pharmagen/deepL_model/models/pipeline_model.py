import os
import glob
import sys
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from keras import layers, Model, Sequential
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. Configuración de paths y carga de archivos ---
os.chdir("C:/Users/rukia/Documents/Master/Pharmagen/deepL_model/docs_data/train_csv")

pharmagen_path = "C:/Users/rukia/Documents/Master"
sys.path.insert(0, pharmagen_path)

fa_ann_path = "C:/Users/rukia/Documents/Master/Pharmagen/deepL_model/models"
csv_files = glob.glob(os.path.join(fa_ann_path, "*.csv"))

# --- 2. Construcción del DataFrame unificado ---
df = pd.concat([pd.read_csv(f, sep=";") for f in csv_files], ignore_index=True)

# --- 3. Carga de diccionarios JSON ---
with open("ATC_farmaco(ENG_dict).json", "r", encoding="utf-8") as f:
    atc_dict = {}
    for entry in json.load(f):
        atc_dict.update(entry)

with open("drug_gene_output.json", "r", encoding="utf-8") as f:
    drug_gene_dict = {}
    for entry in json.load(f):
        drug_gene_dict.update(entry)

# --- 4. Normalización de nombres y limpieza ---
df.columns = [col.strip() for col in df.columns]
df["Drug"] = df["Drug"].astype(str).str.strip().str.upper()
df["Gene"] = df["Gene"].astype(str).str.strip().str.upper()
df["Genotype"] = df["Genotype"].astype(str).str.strip().str.upper()
df["Outcome"] = df["Outcome"].astype(str).str.strip()
df["Effect"] = df["Effect"].astype(str).str.strip()

# --- 5. Añadir columnas útiles ---
def get_atc(drug):
    return next((k for k, v in atc_dict.items() if drug == v.strip().upper()), np.nan)

df["ATC_code"] = df["Drug"].apply(get_atc)
df = df.dropna(subset=["ATC_code"])

# Añadir genes relevantes por fármaco
df["Relevant_Genotypes"] = df["Drug"].map(lambda d: drug_gene_dict.get(d, []))

# --- 6. Definición de targets: fracaso_terapeutico y reacciones_adversas ---
def definir_targets(row):
    fracaso = 1 if (row['Effect'].lower() == 'decreasedefficacy' and 'decreased' in row['Outcome'].lower()) else 0
    reacciones = []
    if row['Outcome'].lower() == 'toxicity': 
        reacciones.append(row['Outcome'])
    return pd.Series({'fracaso_terapeutico': fracaso, 'reacciones_adversas': ','.join(reacciones)})

df[['fracaso_terapeutico', 'reacciones_adversas']] = df.apply(definir_targets, axis=1)

# --- 7. Codificación multihot de features ---
all_drugs = sorted(df["Drug"].unique())
all_genotypes = sorted(df["Genotype"].unique())
all_adverse = sorted(set(','.join(df['reacciones_adversas']).split(',')) - {''})  # Todas las reacciones adversas posibles

def multihot_features(drugs_list, genotypes_list, all_drugs, all_genotypes):
    features = {}
    for drug in all_drugs:
        features[f"drug_{drug}"] = 1 if drug in drugs_list else 0
    for geno in all_genotypes:
        features[f"geno_{geno}"] = 1 if geno in genotypes_list else 0
    return pd.Series(features)

def multilabel_adversas(reacciones, all_adverse):
    # Vector multilabel para cada reacción adversa posible
    values = []
    set_reac = set([r.strip() for r in reacciones.split(',') if r.strip() != ''])
    for adv in all_adverse:
        values.append(1 if adv in set_reac else 0)
    return pd.Series(values)

# Expansión multihot para el DataFrame entero
df_multihot = pd.DataFrame([
    multihot_features([row["Drug"]], [row["Genotype"]], all_drugs, all_genotypes)
    for _, row in df.iterrows()
])
df = pd.concat([df.reset_index(drop=True), df_multihot.reset_index(drop=True)], axis=1)

# Expansión multilabel para reacciones adversas
df_multilabel = pd.DataFrame([
    multilabel_adversas(row['reacciones_adversas'], all_adverse)
    for _, row in df.iterrows()
])
df_multilabel.columns = [f"adv_{adv}" for adv in all_adverse]
df = pd.concat([df.reset_index(drop=True), df_multilabel.reset_index(drop=True)], axis=1)

# --- 8. Features finales y targets ---
X = df[[col for col in df.columns if col.startswith("drug_") or col.startswith("geno_")]].to_numpy()
y = df["fracaso_terapeutico"].to_numpy(dtype=int)
Y_adv = df[[col for col in df.columns if col.startswith("adv_")]].to_numpy(dtype=int)

# --- 9. Train/test split estratificado y sin duplicados cruzados ---
X_train, X_test, y_train, y_test, Yadv_train, Yadv_test = train_test_split(
    X, y, Y_adv, test_size=0.2, random_state=42, stratify=y
)
# Elimina ejemplos de test que estén también en train
train_hashes = set([hash(tuple(row)) for row in X_train])
duplicados = [i for i, row in enumerate(X_test) if hash(tuple(row)) in train_hashes]
if duplicados:
    print(f"Eliminando {len(duplicados)} duplicados de test")
    X_test = np.delete(X_test, duplicados, axis=0)
    y_test = np.delete(y_test, duplicados, axis=0)
    Yadv_test = np.delete(Yadv_test, duplicados, axis=0)

# --- 10. Modelo multisalida (fracaso terapéutico + reacciones adversas) ---
input_feats = Input(shape=(X.shape[1],))
x = Dense(64, activation='relu')(input_feats)
x = Dense(32, activation='relu')(x)

out_fracaso = Dense(1, activation='sigmoid', name="fracaso_terapeutico")(x)
out_adv = Dense(Y_adv.shape[1], activation='sigmoid', name="reacciones_adversas")(x)

model = Model(inputs=input_feats, outputs=[out_fracaso, out_adv])
model.compile(
    optimizer='adam',
    loss={"fracaso_terapeutico": "binary_crossentropy", "reacciones_adversas": "binary_crossentropy"},
    metrics={"fracaso_terapeutico": "accuracy", "reacciones_adversas": "accuracy"}
)

history = model.fit(
    X_train,
    {"fracaso_terapeutico": y_train, "reacciones_adversas": Yadv_train},
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    verbose=str(2)
)

# --- 11. Función input usuario → features multihot para predicción ---
def input_to_multihot(drugs_list, genotypes_list):
    return multihot_features([d.strip().upper() for d in drugs_list],
                             [g.strip().upper() for g in genotypes_list],
                             all_drugs, all_genotypes).to_numpy().reshape(1, -1)

# --- 12. Evaluación y reporte ---
y_pred_fracaso, y_pred_adv = model.predict(X_test)
y_pred_fracaso_bin = (y_pred_fracaso > 0.5).astype(int)
y_pred_adv_bin = (y_pred_adv > 0.5).astype(int)

print("=== FRACASO TERAPÉUTICO ===")
print(classification_report(y_test, y_pred_fracaso_bin, zero_division=0))
print(confusion_matrix(y_test, y_pred_fracaso_bin))

print("\n=== REACCIONES ADVERSAS (multilabel, micro average) ===")
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
print("Hamming loss:", hamming_loss(Yadv_test, y_pred_adv_bin))
print("Micro F1-score:", f1_score(Yadv_test, y_pred_adv_bin, average='micro'))

# Ejemplo de uso de predicción para un caso (descomenta para usar):
# x_new = input_to_multihot(['ASPIRIN'], ['CYP2C19*2'])
# fracaso_pred, adv_pred = model.predict(x_new)
# print('Probabilidad de fracaso terapéutico:', float(fracaso_pred[0][0]))
# adv_labels = [adv for adv, v in zip(all_adverse, adv_pred[0]) if v>0.5]
# print('Reacciones adversas predichas:', adv_labels)

# Guardar el modelo entrenado
model.save("modelo_fracaso_y_adversas.keras")



"""
def predict_fracaso(drugs_list, genotypes_list):
    x = input_to_multihot(drugs_list, genotypes_list)
    prob = model.predict(x)[0][0]
    pred = prob > 0.5
    return {"fracaso_terapeutico_predicho": bool(pred), "probabilidad": float(prob)}

# Ejemplo de uso:
# resultado = predict_fracaso(['ASPIRIN'], ['CYP2C19*2'])
# print(resultado)

# Puedes extender para predecir reacciones adversas usando otros modelos o reglas.
train_hashes = set([hash(tuple(row)) for row in X_train])
test_hashes = set([hash(tuple(row)) for row in X_test])
solapado = train_hashes.intersection(test_hashes)
print(f"Ejemplos idénticos entre train y test: {len(solapado)}")

print(df['fracaso_terapeutico'].value_counts())
"""