import json
import csv
import pandas as pd
import numpy as np
import os
import sys


# Configura tus rutas
name = "var_pheno_ann"
csv_input = str(f"{name}" + "_trimmed.csv")  # El archivo CSV de entrada
csv_output = str(f"{name}" + "_with_ATC.csv")
dict_json = (
    "ATC_farmaco_ENG_corregido.json"  # El archivo JSON con el diccionario key-value
)

# Cargo el diccionario

inversed_dictio = []


with open(dict_json, "r") as f:
    atc_diccionario = json.load(f)
    atc_diccionario = {str(list(entry.values())[0]): entry for entry in atc_diccionario}

for i in atc_diccionario:
    values_keys = {}
    values_keys = dict({v: k for k, v in atc_diccionario[i].items()})
    inversed_dictio.append(values_keys)

pd.DataFrame(inversed_dictio)

"""
    Invertimos el diccionario para que las claves sean los farmacos
    {'CERULETIDE': 'V04CC04'}
    {'METYRAPONE': 'V04CD01'}
    {'CORTICOLIBERIN': 'V04CD04'}
    {'SOMATORELIN': 'V04CD05'}
    {'GALACTOSE': 'V04CE01'}
    """
################################## Hasta aqui todo funciona perfecto --------------------------------
# print(inversed_dictio)

with open(csv_input, "r", encoding="utf-8") as infile, open(
    csv_output, "w", newline="", encoding="utf-8"
) as outfile:
    data_df = pd.read_csv(infile, delimiter=";")

    data_df.insert(0, "ATC", "provis")  # Añade la nueva columna al DataFrame

    for iter in range(len(data_df)):
        drug_name = str(data_df.at[iter, "Drug"])
        for dic in inversed_dictio:
            if drug_name in dic:
                atc_code = dic[drug_name]
                data_df.at[iter, "ATC"] = (
                    atc_code  # Asigna el código ATC a la nueva columna
                )
                break
    data_df.to_csv(outfile, index=False, sep=";", encoding="utf-8")
