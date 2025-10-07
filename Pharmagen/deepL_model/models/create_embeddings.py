import pandas as pd
import numpy as np
import tensorflow as tf
import keras.layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import json
import pickle
import sys

# =======================
# 1. Carga de archivos
# =======================



# Archivos CSV
csv_files = [
    "var_fa_ann_with_ATC.csv",
    "var_pheno_ann_with_ATC.csv",
    "var_drug_ann_with_ATC.csv"
]

# Carga y concatena CSVs
df_csv = pd.concat([pd.read_csv(f, sep=';') for f in csv_files], ignore_index=True)

# Carga JSON: Drug -> lista de genes asociados (diccionario directo)
with open("drug_gene_output.json", "r", encoding="utf-8") as f:
    drug_gene_dict = json.load(f)

# Carga JSON: ATC -> fármaco (diccionario directo)
with open("ATC_farmaco(ENG_dict).json", "r", encoding="utf-8") as f:
    atc_dict = json.load(f)

# Si existe columna ATC, añade columna Drug_from_ATC (útil si necesitas este dato)
#if "ATC" in df_csv.columns:
    #inverted_atc_dict = {v: k for i in atc_dict for k, v in atc_dict[i].items()}
##    df_csv["Drug_from_ATC"] = df_csv["ATC"].map(atc_dict)

# =======================
# 2. Enriquecimiento: columna de genes asociados (multihot)
# =======================

# Crea la lista de todos los posibles genes del JSON
all_genes_set = set()
for genes in drug_gene_dict.values():
    if isinstance(genes, list):
        all_genes_set.update(str(g) for g in genes)
    elif isinstance(genes, str):
        all_genes_set.add(genes)
        
for genes in df_csv["Genotype"].dropna().unique():
    if isinstance(genes, str):
        for gene in genes.split(','):
            all_genes_set.add(gene.strip())
    

all_genes = sorted(all_genes_set)
gene2idx = {gene: idx for idx, gene in enumerate(all_genes)}
num_genes = len(all_genes)

def drug_to_multihot(drug):
    genes = drug_gene_dict.get(drug, [])
    if not isinstance(genes, list):
        genes = [genes] if genes else []
    multihot = np.zeros(num_genes, dtype=np.float32)
    for gene in genes:
        if gene in gene2idx:
            multihot[gene2idx[gene]] = 1.0
    return multihot

# Aplicar multihot encoding a cada fila
df_csv["Gene_multihot"] = df_csv["Drug"].map(drug_to_multihot)

# =======================
# 3. Selección de columnas
# =======================
#ATC;Drug;Gene;Genotype;Alleles;Outcome;Variation;Variation_1;Sentence;Notes;Population;Ref_Genotype
#ATC;Drug;Gene;Genotype;Alleles;Outcome;Variation;Variation_1;Entity;Sentence;Notes
#ATC;Drug;Gene;Genotype;Alleles;Outcome;Variation;Variation_1;Effect;RAM;Sentence;Notes;Population;Ref_Genotype


cat_cols = [
    "Drug",           # Nombre del fármaco
    "Gene",           # Gen
    "Genotype",       # Genotipo
    "Alleles",        # Alelos
    "Outcome",
    "Variation",      # Variación
    "Variation_1",    # Variación 1
    "Effect",         # Efecto
    "RAM",            # Tipos de efecto
    # multihot de genes asociados se añade aparte
]

# Limpia valores NaN (rellena con string vacío, útil para embeddings categóricos)
for col in cat_cols:
    if col in df_csv.columns:
        df_csv[col] = df_csv[col].astype(str).fillna("")
    else:
        df_csv[col] = ""

# =======================
# 4. Codificación de categorías
# =======================
cat2idx = {col: {v: i for i, v in enumerate(sorted(df_csv[col].unique()))} for col in cat_cols}
for col in cat_cols:
    df_csv[f"{col}_idx"] = df_csv[col].map(cat2idx[col])

'''
target_col = "Outcome"
target2idx = {v: i for i, v in enumerate(sorted(df_csv[target_col].unique()))}
df_csv["target"] = df_csv[target_col].map(target2idx)

target_col_two = "Variation"
target2idx_two = {v: i for i, v in enumerate(sorted(df_csv[target_col_two].unique()))}
'''
target_col = "Outcome"
target_col_two = "Variation"
target_pairs = df_csv[[target_col, target_col_two]].apply(lambda row: (row[target_col], row[target_col_two]), axis=1)
target2idx = {v: i for i, v in enumerate(sorted(set(target_pairs)))}
outcome2idx = {v: i for i, v in enumerate(sorted(df_csv[target_col].unique()))}
variation2idx = {v: i for i, v in enumerate(sorted(df_csv[target_col_two].unique()))}
df_csv["outcome"] = df_csv["Outcome"].map(outcome2idx)
df_csv["variation"] = df_csv["Variation"].map(variation2idx)
df_csv["target"] = target_pairs.map(target2idx)

# Eliminar clases con solo 1 muestra para split estratificado
counts = df_csv["target"].value_counts()
if (counts < 2).any():
    print("Clases con solo 1 muestra detectadas y eliminadas:", counts[counts < 2])
df_csv = df_csv[df_csv["target"].isin(counts[counts > 1].index)]

num_outcome_classes = df_csv["outcome"].nunique()

num_variation_classes = df_csv["variation"].value_counts()

# =======================
# 5. Preparación de datos
# =======================

# Extrae la matriz multihot de genes asociados, asegurando tipo y shape correcto
gene_multihot_matrix = np.stack(df_csv["Gene_multihot"].tolist())

X_cat = df_csv[[f"{col}_idx" for col in cat_cols]].to_numpy()
y = df_csv["target"].to_numpy()

# El input completo será (cat features, multihot genes)
X_train_cat, X_test_cat, gene_train, gene_test, y_train, y_test = train_test_split(
    X_cat, gene_multihot_matrix, y, test_size=0.2, random_state=42, stratify=y
)

X_train = [X_train_cat[:, i] for i in range(X_train_cat.shape[1])] + [gene_train]
X_test = [X_test_cat[:, i] for i in range(X_test_cat.shape[1])] + [gene_test]

# =======================
# 6. Modelo de embeddings con multihot
# =======================

embedding_dims = 8
inputs = []
embeddings = []

# Entradas categóricas (embedding)
for i, col in enumerate(cat_cols):
    input_i = keras.layers.Input(shape=(1,), name=f"{col}_idx")
    vocab_size = len(cat2idx[col])
    emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dims, name=f"{col}_emb")(input_i)
    emb = keras.layers.Flatten()(emb)
    inputs.append(input_i)
    embeddings.append(emb)

# Entrada multihot para genes asociados
input_genes = keras.layers.Input(shape=(num_genes,), name="gene_multihot")
# Puedes usar una capa densa para transformar la multihot en un "embedding"
gene_emb = keras.layers.Dense(embedding_dims, activation="relu", name="gene_multihot_emb")(input_genes)
embeddings.append(gene_emb)
inputs.append(input_genes)
'''
x = keras.layers.Concatenate()(embeddings)
output1 = keras.layers.Dense(num_outcome_classes, activation="softmax", name="outcome")(x)
output2 = keras.layers.Dense(num_variation_classes, activation="softmax", name="variation")(x)
output = keras.layers.Dense(len(target2idx), activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=[output1, output2])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
'''
X_train = keras.layers.Concatenate()(embeddings)
X_train = keras.layers.Dense(64, activation="relu")(X_train)
# El modelo tiene dos salidas (un softmax por variable)
output1 = keras.layers.Dense(16, activation="softmax", name="outcome")(X_train)
output2 = keras.layers.Dense(32, activation="softmax", name="variation")(X_train)
model = keras.Model(inputs=inputs, outputs=[output1, output2])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# =======================
# 7. EarlyStopping y entrenamiento
# =======================
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

''' Depuración '''
n_train = len(X_train[0])
n_val = int(n_train * 0.2)
print(f"Entrenamiento: {n_train} - Validación: {n_val}")
for i, arr in enumerate(X_train):
    print(f"Input {i} shape:", arr.shape)
print("y_train shape:", y_train.shape)

model.fit(
    X_train,
    y_train,
    epochs=50,
    validation_split=0.2,
    batch_size=32,
)

# =======================
# 8. Evaluación y guardado
# =======================
loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.3f}")

model.save("modelo_genes_drugs_emb.keras")
with open("cat2idx.pkl", "wb") as f:
    pickle.dump(cat2idx, f)
with open("target2idx.pkl", "wb") as f:
    pickle.dump(target2idx, f)
with open("gene2idx.pkl", "wb") as f:
    pickle.dump(gene2idx, f)
    
# =======================
# 9. Métricas avanzadas
# =======================

from sklearn.metrics import confusion_matrix, classification_report

# Predicciones
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(cm)

# Reporte de clasificación
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred, digits=3))

# (Opcional) Guardar el reporte
with open("classification_report.txt", "w") as f:
    f.write(str(classification_report(y_test, y_pred, digits=4)))

# (Opcional) Mostrar el índice de cada clase
idx2target = {v: k for k, v in target2idx.items()}
print("Índice de clase -> (Outcome, Variation):")
for idx, pair in idx2target.items():
    print(f"{idx}: {pair}")