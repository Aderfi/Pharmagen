import os
import glob
import sys
import pandas as pd
import json
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

os.chdir("C:/Users/rukia/Documents/Master/Pharmagen/deepL_model/docs_data/train_csv")

pharmagen_path = str("C:/Users/rukia/Documents/Master")
Pharmagen = sys.path.insert(0, pharmagen_path)
import Pharmagen as Pharmagen
from Pharmagen import __all__

fa_ann_path = "C:/Users/rukia/Documents/Master/Pharmagen/deepL_model/docs_data/train_csv"
files = os.listdir(fa_ann_path)
fa_ann = os.path.join(fa_ann_path, "*.csv")


# --- 1. Cargar datos y diccionarios ---
df = pd.concat([pd.read_csv(f, sep=";") for f in glob.glob(fa_ann)], ignore_index=True)

# Cargar diccionario ATC <-> farmaco
with open("ATC_farmaco(ENG_dict).json", "r", encoding="utf-8") as f:
    atc_dict = {}
    for entry in json.load(f):
        atc_dict.update(entry)

# Cargar diccionario Drug -> Genotypes relevantes
with open("drug_gene_output.json", "r", encoding="utf-8") as f:
    drug_gene_dict = {}
    for entry in json.load(f):
        drug_gene_dict.update(entry)

# --- 2. Normalizar y mapear nombres de farmacos a ATC (agrega columna ATC_code) ---
def get_atc(drug):
    for k, v in atc_dict.items():
        if drug == v:
            return k
    return np.nan

df["ATC_code"] = df["Drug"].apply(get_atc)
df = df.dropna(subset=["ATC_code"])

# --- 3. Añadir genotipos relevantes por farmaco ---
df["Relevant_Genotypes"] = df["Drug"].apply(lambda d: drug_gene_dict.get(d.strip().upper(), []))

# --- 4. Codificación de variables ---
le_drug = LabelEncoder().fit(df["Drug"])
le_gene = LabelEncoder().fit(df["Gene"])
le_genotype = LabelEncoder().fit(df["Genotype"].astype(str))
le_outcome = LabelEncoder().fit(df["Outcome"].astype(str))
le_effect = LabelEncoder().fit(df["Effect"].astype(str))

df["Drug_idx"] = le_drug.transform(df["Drug"])
df["Gene_idx"] = le_gene.transform(df["Gene"])
df["Genotype_idx"] = le_genotype.transform(df["Genotype"].astype(str))
df["Outcome_idx"] = le_outcome.transform(df["Outcome"].astype(str))
df["Effect_idx"] = le_effect.transform(df["Effect"].astype(str))

# --- 5. Features y targets (dos salidas) ---
X = df[["Drug_idx", "Gene_idx", "Genotype_idx"]].values
y_outcome = df["Outcome_idx"].values
y_effect = df["Effect_idx"].values

# --- 6. Entrenamiento ---
X_train, X_test, y_outcome_train, y_outcome_test, y_effect_train, y_effect_test = train_test_split(
    X, y_outcome, y_effect, test_size=0.2, random_state=42
)

input_drug = keras.layers.Input(shape=(1,), name="drug")
input_gene = keras.layers.Input(shape=(1,), name="gene")
input_geno = keras.layers.Input(shape=(1,), name="genotype")

emb_drug = keras.layers.Embedding(len(le_drug.classes_), 16)(input_drug)
emb_gene = keras.layers.Embedding(len(le_gene.classes_), 16)(input_gene)
emb_geno = keras.layers.Embedding(len(le_genotype.classes_), 16)(input_geno)

x = keras.layers.Concatenate()([emb_drug, emb_gene, emb_geno])
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(32, activation="relu")(x)
x = keras.layers.Dense(16, activation="relu")(x)

out_outcome = keras.layers.Dense(len(le_outcome.classes_), activation="softmax", name="outcome")(x)
out_effect = keras.layers.Dense(len(le_effect.classes_), activation="softmax", name="effect")(x)

model = keras.Model([input_drug, input_gene, input_geno], [out_outcome, out_effect])
model.compile(
    optimizer="adam",
    loss={"outcome": "sparse_categorical_crossentropy", "effect": "sparse_categorical_crossentropy"},
    metrics={"outcome": "accuracy", "effect": "accuracy"}
)

model.fit(
    [X_train[:,0], X_train[:,1], X_train[:,2]],
    {"outcome": y_outcome_train, "effect": y_effect_train},
    validation_split=0.2,
    epochs=30,
    batch_size=16
)

# --- 7. Inferencia y recuperación de notas ---
def predict(drug, gene, genotype):
    di = int(le_drug.transform([drug])[0])
    gi = int(le_gene.transform([gene])[0])
    gti = int(le_genotype.transform([str(genotype)])[0])
    pred_outcome, pred_effect = model.predict([[di], [gi], [gti]])
    idx_outcome = np.argmax(pred_outcome)
    idx_effect = np.argmax(pred_effect)
    outcome = le_outcome.inverse_transform([idx_outcome])[0]
    effect = le_effect.inverse_transform([idx_effect])[0]
    matches = df[
        (df["Drug"] == drug) & (df["Gene"] == gene) & (df["Genotype"].astype(str) == str(genotype))
    ]
    notes, sentence = None, None
    if not matches.empty:
        row = matches.iloc[0]
        notes = row["Notes"]
        sentence = row["Sentence"]
    return outcome, effect, notes, sentence

# Ejemplo de uso:
# outcome, effect, notes, sentence = predict("SITAGLIPTIN", "DPP4", "rs2909451")
# print("Outcome:", outcome, "Effect:", effect, "\nNotes:", notes, "\nSentence:", sentence)