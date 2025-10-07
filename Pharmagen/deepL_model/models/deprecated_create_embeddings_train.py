import pandas as pd
import numpy as np
import tensorflow as tf
import keras.layers
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# pharmagen_path = str(Path(Path(__file__).resolve().parent.parent.parent.parent / "Pharmagen"))
pharmagen_path = str("C:/Users/rukia/Documents/Master")

Pharmagen = sys.path.insert(0, pharmagen_path)
import Pharmagen as Pharmagen
from Pharmagen.config import MODEL_TRAIN_DATA, MODELS_DIR

# RUTAS Y ARCHIVOS
var_drugs_csv = str("var_drug_ann_with_ATC.csv")
path_train_data_path = str(MODEL_TRAIN_DATA)

file = Path(path_train_data_path, var_drugs_csv)

model_files_training = str(MODELS_DIR)

###############################
# Lista de tus archivos CSV
csv_files = ["var_fa_ann_with_ATC.csv", "var_pheno_ann.csv", "drug_gene_output.csv"]

# Define las columnas comunes que quieres usar
columnas_comunes = ["Drug", "Gene", "Variant/Haplotypes", "Alleles", "Outcome", "Effect"]  # ajusta según tus necesidades

# Lee y combina solo las columnas comunes de cada CSV
df = pd.concat([pd.read_csv(f, sep=';')[columnas_comunes] for f in csv_files], ignore_index=True)

# Ahora df tiene solo las columnas comunes de todos los CSV y puedes seguir con el preprocesamiento y entrenamiento
print(df.head())
###############################


# 1. Cargar y limpiar los datos
df = pd.read_csv("var_drug_ann_with_ATC.csv", sep=';')

# 2. Seleccionar columnas a convertir en embeddings
cat_cols = ["Drug", "Gene", "Genotype", "Alleles", "Outcome", "Effect"]

# 3. Codificar categorías como índices
cat2idx = {col: {v: i for i, v in enumerate(df[col].astype(str).unique())} for col in cat_cols}
for col in cat_cols:
    df[f"{col}_idx"] = df[col].astype(str).map(cat2idx[col])

# 4. Definir la variable target (por ejemplo, Effect)
# Puedes cambiar esto por Outcome o lo que prefieras
target_col = "Effect"
target2idx = {v: i for i, v in enumerate(df[target_col].astype(str).unique())}
df["target"] = df[target_col].astype(str).map(target2idx)

# 5. Preparar arrays de entrada y salida
X = df[[f"{col}_idx" for col in cat_cols]].to_numpy()
y = df["target"].to_numpy()

# 6. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Crear modelo de embeddings en TensorFlow
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

x = keras.layers.Concatenate()(embeddings)
x = keras.layers.Dense(16, activation="relu")(x)
x = keras.layers.Dense(8, activation="relu")(x)
output = keras.layers.Dense(len(target2idx), activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 8. Entrenar modelo
model.fit(
    [X_train[:, i] for i in range(len(cat_cols))],
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=4
)

# 9. Evaluar modelo
loss, acc = model.evaluate([X_test[:, i] for i in range(len(cat_cols))], y_test)
print(f"Test accuracy: {acc:.3f}")

# 10. Guardar el modelo y los diccionarios de índices para uso futuro
model.save("modelo_genes_drugs_emb.keras")
import pickle
with open("cat2idx.pkl", "wb") as f:
    pickle.dump(cat2idx, f)
with open("target2idx.pkl", "wb") as f:
    pickle.dump(target2idx, f)