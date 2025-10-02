import sys
from pathlib import Path

# --- Importa rutas y metadatos desde config.py ---
from Pharmagen.config import (
    MODELS_DIR,
    VOCABS_DIR,
    RAW_DATA_DIR,
)
import os
import pandas as pd
from Pharmagen.src.data_handle.preprocess_model import PharmagenPreprocessor
from keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

def main():
    # --- 1. Carga de datos ---
    data_file = RAW_DATA_DIR / "VIH_Genes_Dataframe.csv"  # Cambia aquí si tienes otro archivo
    if not data_file.exists():
        print(f"Archivo de datos no encontrado en: {data_file}")
        sys.exit(1)
    df = pd.read_csv(data_file)
    print(f"Datos cargados desde {data_file}")

    # --- 2. Preprocesamiento/tokenización ---
    preproc = PharmagenPreprocessor()
    X_mut, X_drug = preproc.fit_transform(df, "mutaciones", "medicamentos")
    y = df["outcome"].values  # Cambia 'outcome' si tienes otra columna objetivo

    # --- 3. Split de datos ---
    Xm_train, Xm_test, Xd_train, Xd_test, y_train, y_test = train_test_split(
        X_mut, X_drug, y, test_size=0.2, random_state=42
    )

    # --- 4. Definición del modelo ---
    mut_input = Input(shape=(X_mut.shape[1],), name="mutaciones")
    drug_input = Input(shape=(X_drug.shape[1],), name="medicamentos")

    mut_embed = Embedding(len(preproc.mut_vocab) + 1, 32, mask_zero=True, name="mut_embed")(mut_input)
    mut_embed = Flatten()(mut_embed)

    drug_embed = Embedding(len(preproc.drug_vocab) + 1, 32, mask_zero=True, name="drug_embed")(drug_input)
    drug_embed = Flatten()(drug_embed)

    x = Concatenate()([mut_embed, drug_embed])
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[mut_input, drug_input], outputs=output)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    # --- 5. Callbacks ---
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(VOCABS_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True, monitor="val_loss"),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / "modelo_pharmagen_best.h5"),
            save_best_only=True,
            monitor="val_loss"
        )
    ]

    # --- 6. Entrenamiento ---
    history = model.fit(
        [Xm_train, Xd_train], y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks
    )

    # --- 7. Evaluación ---
    loss, acc = model.evaluate([Xm_test, Xd_test], y_test)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

    # --- 8. Guardado del modelo final y vocabularios ---
    model.save(MODELS_DIR / "modelo_pharmagen_final.h5")
    preproc.save_vocabs(
        str(VOCABS_DIR / "mut_vocab.json"),
        str(VOCABS_DIR / "drug_vocab.json")
    )
    print(f"Modelo y vocabularios guardados en {MODELS_DIR} y {VOCABS_DIR}")

if __name__ == "__main__":
    main()