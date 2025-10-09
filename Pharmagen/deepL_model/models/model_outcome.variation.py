import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras.metrics import SparseTopKCategoricalAccuracy
import joblib
import json
import random
import os

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# === CONFIGURACIÓN ===
INPUT_CSV = "train_data/var_pheno_ann_with_ATC.csv"  # Ajusta el nombre de tu archivo
OUTPUT_MODEL_PATH = "modelo_fracaso_terapeutico.keras"
OUTPUT_ENCODERS_PATH = "label_encoders/"
LOGS_PATH = "logs/entrenamiento_log.json"
RISK_CLASSES = [
    # Ejemplo: 'Fracaso|MutA|MutB|Efecto1'
    # Añade aquí las combinaciones consideradas de riesgo
    'Toxicity|INCREASED|RISK|SIDE_EFFECT',
    'Toxicity|INCREASED|LIKELIHOOD'
]
RAM_COLUMN = "RAM"  # Cambia si el nombre es diferente

# === CARGA DE DATOS DESDE CSV ===
df = pd.read_csv(INPUT_CSV, sep=";", dtype=str)
df = df.reset_index(drop=True)

# === CARGA DE DATOS DESDE JSON ===
with open('train_data/ATC_farmaco(ENG_dict).json', 'r', encoding='utf-8') as f:
    atc_to_farmaco = json.load(f)

with open('train_data/drug_gene_output.json', 'r', encoding='utf-8') as f:
    drug_to_genotype = json.load(f)

rows = []
for atc, drug in atc_to_farmaco.items():
    genotypes = drug_to_genotype.get(drug, [])
    for genotype in genotypes:
        rows.append({"ATC": atc, "Drug": drug, "Genotype": genotype})

df_json = pd.DataFrame(rows)

# === ASEGURA QUE LAS COLUMNAS SEAN IGUALES EN AMBOS DATAFRAMES ===
# Añade columnas que falten en df_json usando valores nulos o por defecto
for col in df.columns:
    if col not in df_json.columns:
        df_json[col] = None

# Opcional: ordena las columnas para que coincidan
df_json = df_json[df.columns]

# === CONCATENA AMBOS DATAFRAMES ===
df_full = pd.concat([df, df_json], ignore_index=True)
df_full = df_full.reset_index(drop=True)

# === AHORA USA df_full PARA TODO EL RESTO DEL PIPELINE ===
# Ejemplo de comprobación
print(df_full.head())

'''
# --- 4. Normalización de nombres y limpieza ---
df.columns = [col.strip() for col in df.columns]
df["Drug"] = df["Drug"].astype(str).str.strip().str.upper()
df["Genotype"] = df["Genotype"].astype(str).str.strip().str.upper()
df["Outcome"] = df["Outcome"].astype(str).str.strip()
df["Variation"] = df["Variation"].astype(str).str.strip().str.upper()
df["Variation_1"] = df["Variation_1"].astype(str).str.strip().str.upper()
df["Effect"] = df["Effect"].astype(str).str.strip()
'''

# === CREACIÓN DE TARGET COMBINADO (como antes) ===
for col in ["Outcome", "Variation", "Variation_1", "Effect"]:
    if col not in df_full.columns:
        raise ValueError(f"Columna {col} no encontrada en los datos combinados.")

df_full["target"] = df_full["Outcome"].astype(str) + "|" + df_full["Variation"].astype(str) + "|" + df_full["Variation_1"].astype(str) + "|" + df_full["Effect"].astype(str)

# === PREPROCESADO DE INPUTS ===
input_cols = ["Drug", "Genotype"]  # Cambia según tus variables de entrada
for col in input_cols:
    if col not in df.columns:
        raise ValueError(f"Columna de input {col} no encontrada en el CSV.")

X = df_full[input_cols]
y = df_full["target"]

# LabelEncoder SOLO para los inputs y el target (para usar índices en Embedding)
encoders = {}
X_idx = pd.DataFrame()
input_cardinalities = []
for col in input_cols:
    le = LabelEncoder()
    X_idx[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le
    input_cardinalities.append(len(le.classes_))

# === FILTRAR CLASES CON SOLO 1 EJEMPLO ===
y_counts = y.value_counts()
mask = y.map(y_counts) > 1
X_idx = X_idx[mask].reset_index(drop=True)
y_filtered = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

# Codificador para el target, para sparse_categorical_crossentropy
target_le = LabelEncoder()
y_encoded = target_le.fit_transform(y_filtered)
encoders["target"] = target_le

os.makedirs(OUTPUT_ENCODERS_PATH, exist_ok=True)
for name, enc in encoders.items():
    joblib.dump(enc, os.path.join(OUTPUT_ENCODERS_PATH, f"{name}_encoder.pkl"))

# === SPLIT ESTRATIFICADO ===
X_train, X_test, y_train, y_test = train_test_split(
    X_idx, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

# === MODELO CON EMBEDDINGS ===
embedding_dims = 32  # Puedes ajustar este valor

inputs = []
embeddings = []
for i, col in enumerate(input_cols):
    input_i = Input(shape=(1,), name=col)
    embedding_i = Embedding(
        input_dim=input_cardinalities[i],
        output_dim=embedding_dims,
        name=f"emb_{col}"
    )(input_i)
    embedding_i = Flatten()(embedding_i)
    inputs.append(input_i)
    embeddings.append(embedding_i)

x = Concatenate()(embeddings)
x = Dense(64, activation='relu')(x)
x = Dropout(0.4)(x)  # Más regularización para evitar overfitting
x = Dense(32, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(len(np.unique(y_encoded)), activation='softmax')(x)

model = Model(inputs=inputs, outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', SparseTopKCategoricalAccuracy(k=3)]
)

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Prepara los datos para el modelo funcional (list of arrays)
def df_to_input_list(df_):
    return [df_[col].values for col in input_cols]

X_train_list = df_to_input_list(X_train)
X_test_list = df_to_input_list(X_test)

history = model.fit(
    X_train_list, y_train, 
    validation_split=0.15, 
    epochs=200, 
    batch_size=32, 
    callbacks=[es],
    verbose=1 # type: ignore
)

# === EVALUACIÓN ===
loss, accuracy, top3 = model.evaluate(X_test_list, y_test, verbose=0) # type: ignore
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}, Test top-3 acc: {top3:.4f}")
print("Nº de clases únicas:", len(set(y_encoded))) # type: ignore

# === GUARDADO DE MODELO Y LOGS ===
os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
model.save(OUTPUT_MODEL_PATH)
with open(LOGS_PATH, "w") as f:
    json.dump({
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "test_top3_accuracy": float(top3),
        "input_cols": input_cols,
        "target_classes": list(target_le.classes_),
        "num_classes": len(target_le.classes_),
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "input_cardinalities": input_cardinalities
    }, f, indent=2)

'''
# === BLOQUE DE PREDICCIÓN Y LÓGICA DE "FRACASO TERAPÉUTICO" ===
def predict_fracaso_terapeutico(input_farmaco, input_mutacion, risk_classes=RISK_CLASSES):
    # Cargar modelo y encoders
    model = keras.models.load_model(OUTPUT_MODEL_PATH)
    farmaco_le = joblib.load(os.path.join(OUTPUT_ENCODERS_PATH, "farmaco_encoder.pkl"))
    mutacion_le = joblib.load(os.path.join(OUTPUT_ENCODERS_PATH, "mutacion_encoder.pkl"))
    target_le = joblib.load(os.path.join(OUTPUT_ENCODERS_PATH, "target_encoder.pkl"))
    
    # Codificar inputs
    X_pred = [np.array([farmaco_le.transform([input_farmaco])[0]]), np.array([mutacion_le.transform([input_mutacion])[0]])]
    
    # Predicción
    pred_proba = model.predict(X_pred)
    pred_class_idx = np.argmax(pred_proba, axis=1)[0]
    pred_class = target_le.inverse_transform([pred_class_idx])[0]
    
    # Lógica de fracaso terapéutico
    fracaso = pred_class in risk_classes
    detalle = {"combinacion_predicha": pred_class}
    
    # Buscar RAM asociado si aplica
    if fracaso:
        filas_match = df[df["target"] == pred_class]
        detalle["RAM"] = list(filas_match[RAM_COLUMN].unique())
    return {"fracaso_terapeutico": fracaso, "detalle": detalle}

# === EJEMPLO DE USO DEL BLOQUE DE PREDICCIÓN ===
if __name__ == "__main__":
    ejemplo_farmaco = df.iloc[0][input_cols[0]]
    ejemplo_mutacion = df.iloc[0][input_cols[1]]
    resultado = predict_fracaso_terapeutico(ejemplo_farmaco, ejemplo_mutacion)
    print("Predicción ejemplo:", resultado)
'''