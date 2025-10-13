import torch
import pandas as pd
import joblib
from pathlib import Path
from . import model_structure 
from model_structure import PharmacoModel
from src.config.config import *

'''--------------------------<<< DOC STRING >>>--------------------------
Función para predecir usando el modelo entrenado previamente con los encoders
y un DataFrame de entrada con las columnas 'Drug' y 'Genotype', en su defecto
se pueden usar dos listas independientes en formato txt con los nombres de 
los fármacos y genotipos a predecir.

'''
 
def predict_model_Outc_Var(PREDICTION_INPUT, SAVE_MODEL_AS, SAVE_ENCODERS_AS, RESULTS_DIR, equivalencias):
    
    # Parámetros del modelo (Calculados mediante entrenamiento con Optuna)
    EMB_DIM = 64
    HIDDEN_DIM = 704

    print(f"\n  Loading model and encoders... \
        \n          Model: PGen Outcome_Variation \
        \n          Encoders: {SAVE_ENCODERS_AS}")    
    
    # Cargar encoders para conocer dimensiones
    encoders = joblib.load(SAVE_ENCODERS_AS)
    n_drugs = len(encoders['Drug'].classes_)
    n_genotypes = len(encoders['Genotype'].classes_)
    n_effects = len(encoders['Outcome'].classes_)
    n_outcomes = len(encoders['Variation'].classes_)
    
    
    # PREDICCIÓN A PARTIR DE ENTRADAS DEFINIDAS
    print("\n======== PREDICCIONES =========")
    # Cargar modelo y encoders (para asegurar predicción tras entrenamiento)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders = joblib.load(SAVE_ENCODERS_AS)
    model = PharmacoModel(n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(SAVE_MODEL_AS, map_location=torch.device('cpu')))
    model.eval()

    # Normalizar y codificar entradas
    print("\n Normalizando y codificando entradas...")
    input_df = pd.DataFrame(PREDICTION_INPUT, columns=['Drug', 'Genotype'], dtype=str, index=None)


    # Normalizar genotipo usando equivalencias
    input_df['Genotype'] = input_df['Genotype'].map(lambda x: equivalencias.get(x, x))
    # Codificar Drug y Genotype
    for col in ['Drug', 'Genotype']:
        if col in input_df:
            # Si el valor no está en los encoders, asigna -1 (desconocido)
            input_df[col] = input_df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
        

    # Filtrar entradas válidas (que se hayan codificado correctamente)
    valid_idx = (input_df['Drug'] != -1) & (input_df['Genotype'] != -1)
    if not valid_idx.any():
        print("No hay entradas válidas para predecir (Drug o Genotype no reconocidos en el modelo). \
            \n Por favor, asegúrese de que el fármaco se encuentra en el vocabulario del modelo. \
            \n Y que no ha habido ninguna errata a la hora de escribirlo.")
    else:
        drug_tensor = torch.tensor(input_df.loc[valid_idx, 'Drug'].values, dtype=torch.long)
        genotype_tensor = torch.tensor(input_df.loc[valid_idx, 'Genotype'].values, dtype=torch.long)
        with torch.no_grad():
            effect_pred, outcome_pred = model(drug_tensor, genotype_tensor)
            effect_labels = torch.argmax(effect_pred, dim=1).numpy()
            outcome_labels = torch.argmax(outcome_pred, dim=1).numpy()
            # Decodificar los resultados
            effect_decoded = encoders['Effect'].inverse_transform(effect_labels)
            outcome_decoded = encoders['Outcome'].inverse_transform(outcome_labels)

        
        with open(str(Path(RESULTS_DIR + "predicciones.txt")), "w", encoding="utf-8") as f: # type: ignore
            for i in range(len(effect_decoded)):
                entrada_codificada = input_df[valid_idx].iloc[i].to_dict()
                # Decodifica Drug y Genotype usando los encoders inversos
                drug_name = encoders['Drug'].inverse_transform([entrada_codificada['Drug']])[0]
                genotype_name = encoders['Genotype'].inverse_transform([entrada_codificada['Genotype']])[0]
                result_str = (
                f"Input: {{'Drug': '{drug_name}', 'Genotype': '{genotype_name}'}}\n"
                f"  Predicted Effect:  {effect_decoded[i]}\n"
                f"  Predicted Outcome: {outcome_decoded[i]}\n"
                + "-" * 40 + "\n"
                )
                print(result_str, end="")  # Muestra también en consola
                f.write(result_str)

        print(f"\nPredicciones guardadas en {RESULTS_DIR}/predicciones.txt")
        
def predict_model_Eff_Entity(PREDICTION_INPUT, SAVE_MODEL_AS, SAVE_ENCODERS_AS, RESULTS_DIR, equivalencias):
    # Parámetros del modelo (deben coincidir con los usados en el entrenamiento)
    EMB_DIM = 16       # Debe coincidir con el valor usado en el entrenamiento
    HIDDEN_DIM = 256   # Debe coincidir con el valor usado en el entrenamiento
    LEARNING_RATE = 0.001  # Debe coincidir con el valor usado en el entrenamiento

    # Cargar encoders para conocer dimensiones
    encoders = joblib.load(SAVE_ENCODERS_AS)
    n_drugs = len(encoders['Drug'].classes_)
    n_genotypes = len(encoders['Genotype'].classes_)
    n_effects = len(encoders['Effect'].classes_)
    n_outcomes = len(encoders['Outcome'].classes_)
    # PREDICCIÓN A PARTIR DE ENTRADAS DEFINIDAS
    print("\n=== PREDICCIONES ===")
    # Cargar modelo y encoders (para asegurar predicción tras entrenamiento)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoders = joblib.load(SAVE_ENCODERS_AS)
    model = PharmacoModel(n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
    model.load_state_dict(torch.load(SAVE_MODEL_AS, map_location=torch.device('cpu')))
    model.eval()

    # Normalizar y codificar entradas
    input_df = pd.DataFrame(PREDICTION_INPUT, columns=['Drug', 'Genotype'], dtype=str, index=None)

    # Normalizar genotipo usando equivalencias
    input_df['Genotype'] = input_df['Genotype'].map(lambda x: equivalencias.get(x, x))
    # Codificar Drug y Genotype
    for col in ['Drug', 'Genotype']:
        if col in input_df:
            # Si el valor no está en los encoders, asigna -1 (desconocido)
            input_df[col] = input_df[col].map(lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1)
        

    # Filtrar entradas válidas (que se hayan codificado correctamente)
    valid_idx = (input_df['Drug'] != -1) & (input_df['Genotype'] != -1)
    if not valid_idx.any():
        print("No hay entradas válidas para predecir (Drug o Genotype no reconocidos en el modelo).")
    else:
        drug_tensor = torch.tensor(input_df.loc[valid_idx, 'Drug'].values, dtype=torch.long)
        genotype_tensor = torch.tensor(input_df.loc[valid_idx, 'Genotype'].values, dtype=torch.long)
        with torch.no_grad():
            effect_pred, outcome_pred = model(drug_tensor, genotype_tensor)
            effect_labels = torch.argmax(effect_pred, dim=1).numpy()
            outcome_labels = torch.argmax(outcome_pred, dim=1).numpy()
            # Decodificar los resultados
            effect_decoded = encoders['Effect'].inverse_transform(effect_labels)
            outcome_decoded = encoders['Outcome'].inverse_transform(outcome_labels)

        
        with open(str(Path(RESULTS_DIR + "predicciones.txt")), "w", encoding="utf-8") as f: # type: ignore
            for i in range(len(effect_decoded)):
                entrada_codificada = input_df[valid_idx].iloc[i].to_dict()
                # Decodifica Drug y Genotype usando los encoders inversos
                drug_name = encoders['Drug'].inverse_transform([entrada_codificada['Drug']])[0]
                genotype_name = encoders['Genotype'].inverse_transform([entrada_codificada['Genotype']])[0]
                result_str = (
                f"Input: {{'Drug': '{drug_name}', 'Genotype': '{genotype_name}'}}\n"
                f"  Predicted Effect:  {effect_decoded[i]}\n"
                f"  Predicted Outcome: {outcome_decoded[i]}\n"
                + "-" * 40 + "\n"
                )
                print(result_str, end="")  # Muestra también en consola
                f.write(result_str)

        print(f"\nPredicciones guardadas en {RESULTS_DIR}/predicciones.txt")
