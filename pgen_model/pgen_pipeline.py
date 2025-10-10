import os
import sys
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path


# --- 1. CONFIGURACIÓN GENERAL (AJUSTAR AQUÍ) ---
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# === RUTAS PRINCIPALES (MODIFICA SEGÚN TU ESTRUCTURA) ===
DATA_DIR = Path("docs_data")           # Carpeta donde están los CSV/JSON de entrenamiento
MODEL_DIR = Path("models")              # Carpeta donde guardar modelos entrenados
ENCODER_DIR = Path("labels_vocabs")    # Carpeta donde guardar label encoders
LOGS_DIR = Path("../logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ENCODER_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# === ARCHIVOS (MODIFICA LOS NOMBRES SEGÚN TU CASO) ===
ATC_JSON = DATA_DIR / "ATC_drug_dict_ENG.json"   # Diccionario ATC→Fármaco
DRUGGENE_JSON = DATA_DIR / "drug_gene_output.json"   # Diccionario Fármaco→Genotipo

input_csv_list = [f for f in DATA_DIR.glob(f"{DATA_DIR}/*.csv")]

# === COLUMNAS DE ENTRADA Y TARGETS (AJUSTA AQUÍ) ===
INPUT_COLS = ["Drug", "Genotype"]  # Variables de entrada al modelo
#TARGETS = ["fracaso_terapeutico", "reacciones_adversas"]  # Salidas (puedes añadir más)
TARGETS = ["Outcome", "Variation", "Variation_1", "Effect", "Entity"]

# === HIPERPARÁMETROS DE MODELO (AJUSTA AQUÍ) ===
EMBEDDING_DIM = 32
DENSE_UNITS = [64, 32]
DROPOUT_RATES = [0.3, 0.2]
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10
VAL_SPLIT = 0.15
TEST_SIZE = 0.2

# === THRESHOLDS PARA BINARIZAR SALIDAS ===
THRESHOLD_FRACASO = 0.5
THRESHOLD_ADV = 0.5

# --- 2. CARGA Y LIMPIEZA DE DATOS ---
print("Cargando datos...")
df = pd.DataFrame()
for i in range(1, len(input_csv_list)):
    if len(input_csv_list) == 1:
        df = pd.read_csv(input_csv_list[0], sep=";", dtype=str)
        df = df.reset_index(drop=True)
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
    else:
        df_temp = pd.read_csv(input_csv_list[i], sep=";", dtype=str)
        df = df_temp.reset_index(drop=True)
        for col in df.columns:
            df_temp[col] = df_temp[col].astype(str).str.strip().str.upper()
        df = pd.concat([df, df_temp], ignore_index=True)

df['Variation_comb'] = df['Variation'].astype(str) + ' ' + df['Variation_1'].astype(str)

# --- 3. CARGA DE DICCIONARIOS (OPCIONAL, sólo si los necesitas para features extra) ---
if ATC_JSON.exists():
    with open(ATC_JSON, "r", encoding="utf-8") as f:
        atc_to_drug = json.load(f)
else:
    atc_to_drug = {}

if DRUGGENE_JSON.exists():
    with open(DRUGGENE_JSON, "r", encoding="utf-8") as f:
        drug_to_gene = json.load(f)
else:
    drug_to_gene = {}

# --- 4. CREACIÓN DE TARGETS PERSONALIZADOS (puedes modificar la lógica aquí) ---
# Ejemplo: target binario fracaso_terapeutico y multilabel reacciones_adversas
def definir_targets(row):
    fracaso = True if ( \
        row.get('OUTCOME', '') in ['LACK OF EFFICACY', 'THERAPEUTIC FAILURE', 'NO RESPONSE', 'POOR RESPONSE', 'RESISTANCE', 'RELAPSE', 'PROGRESSION', 'DEATH', 'DECEASED', 'DIED'] and
        row.get
            
            
            
        ])
        
        
        
        and 'INCREASED' in row.get('OUTCOME', '')) else False
    reacciones = []
    if row.get('OUTCOME', '') == 'TOXICITY': 
        reacciones.append(row.get('OUTCOME', ''))
    return pd.Series({'fracaso_terapeutico': fracaso, 'reacciones_adversas': ','.join(reacciones)})

if not all([t in df.columns for t in TARGETS]):
    # Si faltan los targets, créalos aquí (modifica según tu lógica)
    df[TARGETS] = df.apply(definir_targets, axis=1)

# --- 5. ENCODING DE VARIABLES CATEGÓRICAS (INPUTS Y TARGETS) ---
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import joblib

encoders = {}

# Inputs: LabelEncoder por variable
X_encoded = pd.DataFrame()
input_cardinalities = []
for col in INPUT_COLS:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    joblib.dump(le, ENCODER_DIR / f"{col}_encoder.pkl")
    input_cardinalities.append(len(le.classes_))

# Targets:
# Fracaso terapéutico → binario, Reacciones adversas → multilabel
y_fracaso = df["fracaso_terapeutico"].astype(int).values
mlb = MultiLabelBinarizer()
y_adv = mlb.fit_transform(df["reacciones_adversas"].str.split(','))
encoders["reacciones_adversas"] = mlb
joblib.dump(mlb, ENCODER_DIR / "reacciones_adversas_encoder.pkl")

# --- 6. SPLIT DE DATOS ---
from sklearn.model_selection import train_test_split

# Elimina clases con <2 ejemplos para evitar problemas de split
mask = pd.Series(y_fracaso).map(pd.Series(y_fracaso).value_counts()) > 1
X_encoded = X_encoded[mask].reset_index(drop=True)
y_fracaso = y_fracaso[mask]
y_adv = y_adv[mask]

X_train, X_test, y_train, y_test, Yadv_train, Yadv_test = train_test_split(
    X_encoded.values, y_fracaso, y_adv, test_size=TEST_SIZE, random_state=SEED, stratify=y_fracaso
)

# --- 7. DEFINICIÓN DEL MODELO (AJUSTA ARQUITECTURA AQUÍ) ---
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout

inputs = []
embeddings = []
for i, col in enumerate(INPUT_COLS):
    input_i = Input(shape=(1,), name=col)
    embedding_i = Embedding(
        input_dim=input_cardinalities[i],
        output_dim=EMBEDDING_DIM,
        name=f"emb_{col}"
    )(input_i)
    embedding_i = Flatten()(embedding_i)
    inputs.append(input_i)
    embeddings.append(embedding_i)
x = Concatenate()(embeddings)
for units, drop in zip(DENSE_UNITS, DROPOUT_RATES):
    x = Dense(units, activation='relu')(x)
    x = Dropout(drop)(x)

out_fracaso = Dense(1, activation='sigmoid', name="fracaso_terapeutico")(x)
out_adv = Dense(y_adv.shape[1], activation='sigmoid', name="reacciones_adversas")(x)
model = Model(inputs=inputs, outputs=[out_fracaso, out_adv])

model.compile(
    optimizer='adam',
    loss={"fracaso_terapeutico": "binary_crossentropy", "reacciones_adversas": "binary_crossentropy"},
    metrics={"fracaso_terapeutico": "accuracy", "reacciones_adversas": "accuracy"}
)
model.summary()

# --- 8. ENTRENAMIENTO ---
from keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
    ModelCheckpoint(filepath=str(MODEL_DIR / "modelo_pharmagen_best.h5"), save_best_only=True, monitor="val_loss")
]

X_train_list = [X_train[:, i] for i in range(X_train.shape[1])]
X_test_list = [X_test[:, i] for i in range(X_test.shape[1])]

history = model.fit(
    X_train_list,
    {"fracaso_terapeutico": y_train, "reacciones_adversas": Yadv_train},
    validation_split=VAL_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose='1'
)

# --- 9. EVALUACIÓN ---
score = model.evaluate(X_test_list, {"fracaso_terapeutico": y_test, "reacciones_adversas": Yadv_test}, verbose='1')
print(f"Test Score: {score}")

from sklearn.metrics import classification_report, hamming_loss, f1_score

y_pred_fracaso, y_pred_adv = model.predict(X_test_list)
y_pred_fracaso_bin = (y_pred_fracaso > THRESHOLD_FRACASO).astype(int)
y_pred_adv_bin = (y_pred_adv > THRESHOLD_ADV).astype(int)

print("\n=== Reporte fracaso_terapeutico ===")
print(classification_report(y_test, y_pred_fracaso_bin, zero_division=0))
print("\n=== Reacciones adversas (multilabel, micro average) ===")
print("Hamming loss:", hamming_loss(Yadv_test, y_pred_adv_bin))
print("Micro F1-score:", f1_score(Yadv_test, y_pred_adv_bin, average='micro'))

# --- 10. GUARDADO DE MODELO Y LOGS ---
model.save(MODEL_DIR / "modelo_pharmagen_final.h5")
with open(LOGS_DIR / "entrenamiento_log.json", "w") as f:
    json.dump({
        "test_score": [float(s) for s in score],
        "input_cols": INPUT_COLS,
        "num_classes_adv": y_adv.shape[1],
        "training_size": int(len(X_train)),
        "test_size": int(len(X_test))
    }, f, indent=2)

print("\n=== Pipeline completado. Modelo y encoders guardados. ===")

# --- 11. FUNCIONES ÚTILES PARA FUTURO USO/PREDICCIÓN ---
def encode_inputs(drug, genotype):
    """Codifica los inputs usando los encoders entrenados (ajusta si cambias INPUT_COLS)."""
    return [
        encoders["Drug"].transform([drug.upper()]),
        encoders["Genotype"].transform([genotype.upper()])
    ]

def predict_case(drug, genotype):
    """Obtiene la predicción binaria para un caso dado."""
    x = encode_inputs(drug, genotype)
    fracaso_pred, adv_pred = model.predict([np.array([x[0]]), np.array([x[1]])])
    return {
        "prob_fracaso": float(fracaso_pred[0][0]),
        "fracaso_predicho": bool(fracaso_pred[0][0] > THRESHOLD_FRACASO),
        "reacciones_adversas_pred": [lab for lab, v in zip(mlb.classes_, adv_pred[0]) if v > THRESHOLD_ADV]
    }

# --- FIN DEL PIPELINE ---