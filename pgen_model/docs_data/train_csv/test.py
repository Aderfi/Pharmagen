import pandas as pd
import json
import re
import numpy as np

# --- 1. Carga de Archivos ---

# Rutas a tus archivos de entrada
ruta_csv = "var_drug_ann_with_ATC.csv"
ruta_json = "ATC_drug_dict_ENG.json"
nombre_columna_farmacos = "Drug"
nombre_columna_editar = "ATC"

# Cargar el diccionario de traducción desde el archivo JSON
try:
    with open(ruta_json, "r", encoding="utf-8") as f:
        diccionario_atc = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo JSON en la ruta: {ruta_json}")
    exit()

# Cargar los datos del CSV en un DataFrame de pandas
try:
    df = pd.read_csv(ruta_csv, sep=";", encoding="utf-8")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo CSV en la ruta: {ruta_csv}")
    exit()

# --- 2. Procesamiento con Bucle 'for' ---

print("Procesando los datos con un bucle for...")

## A10BH01;SITAGLIPTIN;GLP1R;rs3765467;AG;Efficacy;increased response to ⮕ SITAGLIPTIN;Genotype AG is associated with increased response to sitagliptin in people with Diabetes Mellitus, Type 2.;"""Conversely, patients with the rs3765467 AG genotype in the study group demonstrated a median HbA1c improvement of 1.42 (IQR, 1.22Ã‘1.68) compared with 1.08 (IQR, 0.97Ã‘1.15) in the control group (P?=?.023), indicating favorable responses to both treatments."""; in people with Other:Diabetes Mellitus, Type 2;
##        ;IVACAFTOR / LUMACAFTOR;


for (
    ind,
    row,
) in df.iterrows():  # Bucle para procesar columnas donde solo hay 1 medicamento
    farmaco = str(row["Drug"]).strip().upper()
    atc_code = [k for k, v in diccionario_atc.items() if v.strip().upper() == farmaco]

    df.at[ind, "ATC"] = "/".join(
        atc_code
    )  # Asigna el código ATC o NaN si no se encuentra

# --- 3. Procesamiento sin Bucle (usando map) ---
print("Procesando los datos sin bucle (usando map)...")
# Crea una función para mapear los fármacos a sus códigos ATC


def mapear_varios_atc(farmaco):
    farmaco = str(farmaco)
    farmacos_lista = [re.split(r"[, ]", farmaco)][0]
    atc_codes = set()  # Usar un conjunto para evitar duplicados
    for f in farmacos_lista:
        codigos = [k for k, v in diccionario_atc.items() if v == f]
        atc_codes.update(codigos)
    return "/".join(atc_codes)


# Aplica la función de mapeo a la columna 'Drug' y asigna los resultados a la columna 'ATC'
df["ATC"] = df["Drug"].map(mapear_varios_atc)

print(df.head())
# Guardamos el resultado en un nuevo archivo CSV
ruta_salida_csv = "resultados_traducidos_bucle.csv"
df.to_csv(ruta_salida_csv, sep=";", index=False, encoding="utf-8-")

print(f"\n✅ ¡Proceso completado! Resultados guardados en: {ruta_salida_csv}")
