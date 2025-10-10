import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.callbacks import EarlyStopping
import joblib
import json
import random
import os
import sys

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# === CONFIGURACIÓN ===
INPUT_CSV = "train_data/var_pheno_ann_with_ATC.csv"  # Ajusta el nombre de tu archivo
OUTPUT_MODEL_PATH = "modelo_variations_1.keras"
OUTPUT_ENCODERS_PATH = "label_encoders/"
LOGS_PATH = "logs/entrenamiento_log.json"
RISK_CLASSES = [
    # Ejemplo: 'Fracaso|MutA|MutB|Efecto1'
    # Añade aquí las combinaciones consideradas de riesgo
    'Toxicity|INCREASED|RISK|SIDE_EFFECT',
    'Toxicity|INCREASED|LIKELIHOOD'
]
RAM_COLUMN = "RAM"  # Cambia si el nombre es diferente

# === CARGA DE DATOS ===
df = pd.read_csv(INPUT_CSV, sep=";", dtype=str)
df = df.reset_index(drop=True)

# === CREACIÓN DE TARGET COMBINADO ===
for col in ["Variation", "Variation_1"]:
    if col not in df.columns:
        raise ValueError(f"Columna {col} no encontrada en el CSV.")

df["target"] = df["Variation"].astype(str) + "|" + df["Variation_1"].astype(str)

# === PREPROCESADO DE INPUTS ===
# Ajusta según tus columnas de entrada reales:
input_cols = ["Drug", "Genotype"]  # Cambia según tus variables de entrada
for col in input_cols:
    if col not in df.columns:
        raise ValueError(f"Columna de input {col} no encontrada en el CSV.")

X = df[input_cols]
y = df["target"]

# Codificación de variables categóricas (inputs)
encoders = {}
X_encoded = pd.DataFrame()
for col in input_cols:
    le = Embedding()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# === FILTRAR CLASES CON SOLO 1 EJEMPLO ===
y_counts = y.value_counts()
mask = y.map(y_counts) > 1
X_encoded = X_encoded[mask].reset_index(drop=True)
y_filtered = y[mask].reset_index(drop=True)
df = df[mask].reset_index(drop=True)

# Codificador para el target (ahora SÍ)
target_le = Embedding()
y_encoded = target_le.fit_transform(y_filtered)
encoders["target"] = target_le

# Guarda encoders
os.makedirs(OUTPUT_ENCODERS_PATH, exist_ok=True)
for name, enc in encoders.items():
    joblib.dump(enc, os.path.join(OUTPUT_ENCODERS_PATH, f"{name}_encoder.pkl"))

# === SPLIT ESTRATIFICADO ===
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

# === MODELO SIMPLE (puedes mejorar la arquitectura) ===
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train, 
    validation_split=0.15, 
    epochs=200, 
    batch_size=32, 
    callbacks=[es],
    verbose=1 # type: ignore
)

# === EVALUACIÓN ===
loss, accuracy = model.evaluate(X_test, y_test, verbose=str(0))
print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
print("Nº de clases únicas:", len(set(y_encoded))) # type: ignore

# === GUARDADO DE MODELO Y LOGS ===
os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)
model.save(OUTPUT_MODEL_PATH)
with open(LOGS_PATH, "w") as f:
    json.dump({
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "input_cols": input_cols,
        "target_classes": list(target_le.classes_),
        "num_classes": len(target_le.classes_),
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test))
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
    X_pred = np.array([
        [farmaco_le.transform([input_farmaco])[0], mutacion_le.transform([input_mutacion])[0]]
    ])
    
    # Predicción
    pred_proba = model.predict(X_pred)
    pred_class_idx = np.argmax(pred_proba, axis=1)[0]
    pred_class = target_le.inverse_transform([pred_class_idx])[0]
    
    # Lógica de fracaso terapéutico
    fracaso = pred_class in risk_classes
    detalle = {"combinacion_predicha": pred_class}
    
    # Buscar RAM asociado si aplica
    if fracaso:
        # Busca en el df original la fila que coincide y saca el RAM
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