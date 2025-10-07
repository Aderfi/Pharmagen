import pandas as pd
import numpy as np
import tensorflow as tf
import keras.layers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import json
import pickle

# =======================
# 1. Carga de archivos
# =======================

csv_files = [
    "var_fa_ann_long.csv",
    "var_pheno_ann_long.csv",
    "var_drug_ann_long.csv"
]
df_csv = pd.concat([pd.read_csv(f, sep=';') for f in csv_files], ignore_index=True)

with open("drug_gene_output.json", "r", encoding="utf-8") as f:
    drug_gene_dict = json.load(f)
with open("ATC_farmaco(ENG_dict).json", "r", encoding="utf-8") as f:
    atc_dict = json.load(f)

# =======================
# 2. Enriquecimiento: columna de genes asociados (multihot)
# =======================

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

df_csv["Gene_multihot"] = df_csv["Drug"].map(drug_to_multihot)

# =======================
# 3. Selección de columnas
# =======================

cat_cols = [
    "Drug", "Gene", "Genotype", "Alleles", "Outcome",
    "Variation", "Variation_1", "Effect", "RAM"
]

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
num_variation_classes = df_csv["variation"].nunique()

# =======================
# 5. Preparación de datos
# =======================

gene_multihot_matrix = np.stack(df_csv["Gene_multihot"].tolist())
X_cat = df_csv[[f"{col}_idx" for col in cat_cols]].to_numpy()
y_outcome = df_csv["outcome"].to_numpy()
y_variation = df_csv["variation"].to_numpy()

# Split estratificado por el target conjunto
X_train_cat, X_test_cat, gene_train, gene_test, y_train_outcome, y_test_outcome, y_train_variation, y_test_variation = train_test_split(
    X_cat, gene_multihot_matrix, y_outcome, y_variation,
    test_size=0.2, random_state=42, stratify=df_csv["target"]
)

X_train_inputs = [X_train_cat[:, i] for i in range(X_train_cat.shape[1])] + [gene_train]
X_test_inputs = [X_test_cat[:, i] for i in range(X_test_cat.shape[1])] + [gene_test]

# =======================
# 6. Modelo de embeddings con multihot
# =======================

embedding_dims = 8
inputs = []
embeddings = []

for i, col in enumerate(cat_cols):
    input_i = keras.layers.Input(shape=(1,), name=f"{col}_idx")
    vocab_size = len(cat2idx[col])
    emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dims, name=f"{col}_emb")(input_i)
    emb = keras.layers.Flatten()(emb)
    inputs.append(input_i)
    embeddings.append(emb)

input_genes = keras.layers.Input(shape=(num_genes,), name="gene_multihot")
gene_emb = keras.layers.Dense(embedding_dims, activation="relu", name="gene_multihot_emb")(input_genes)
embeddings.append(gene_emb)
inputs.append(input_genes)

x = keras.layers.Concatenate()(embeddings)
x = keras.layers.Dense(64, activation="relu")(x)
output1 = keras.layers.Dense(num_outcome_classes, activation="sigmoid", name="outcome")(x)
output2 = keras.layers.Dense(num_variation_classes, activation="sigmoid", name="variation")(x)
model = keras.Model(inputs=inputs, outputs=[output1, output2])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# =======================
# 7. EarlyStopping y entrenamiento
# =======================
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

model.fit(
    X_train_inputs,
    [y_train_outcome, y_train_variation],
    epochs=50,
    validation_split=0.2,
    batch_size=32,
    callbacks=[early_stop]
)

# =======================
# 8. Evaluación y guardado
# =======================
loss, acc_outcome, acc_variation = model.evaluate(X_test_inputs, [y_test_outcome, y_test_variation])
print(f"Test accuracy outcome: {acc_outcome:.3f}")
print(f"Test accuracy variation: {acc_variation:.3f}")

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
y_pred_probs = model.predict(X_test_inputs)
y_pred_outcome = np.argmax(y_pred_probs[0], axis=1)
y_pred_variation = np.argmax(y_pred_probs[1], axis=1)

# Matriz de confusión y reporte para cada salida
print("Matriz de confusión (outcome):")
print(confusion_matrix(y_test_outcome, y_pred_outcome))
print("Reporte de clasificación (outcome):")
print(classification_report(y_test_outcome, y_pred_outcome, digits=3))

print("Matriz de confusión (variation):")
print(confusion_matrix(y_test_variation, y_pred_variation))
print("Reporte de clasificación (variation):")
print(classification_report(y_test_variation, y_pred_variation, digits=3))

# Guardar reporte
with open("classification_report.txt", "w") as f:
    f.write("OUTCOME REPORT\n")
    f.write(str(classification_report(y_test_outcome, y_pred_outcome, digits=4)))
    f.write("\nVARIATION REPORT\n")
    f.write(str(classification_report(y_test_variation, y_pred_variation, digits=4)))

idx2target = {v: k for k, v in target2idx.items()}
print("Índice de clase -> (Outcome, Variation):")
for idx, pair in idx2target.items():
    print(f"{idx}: {pair}")