import json
import pandas as pd
import numpy as np
import os
import sys
import csv
import re

'''
with open("drug_gene_output.json", "r", encoding="utf-8") as f:
    drug_gene_list = json.load(f)
    drug_gene_dict = {list(d.keys())[0]: list(d.values())[0] for d in drug_gene_list}

print(drug_gene_dict)
i = 0
for i in range(len(drug_gene_dict)):
    print(drug_gene_dict[i])
    i += 1
    if i > 5:
        break   
'''



'''
    Convierte los archivos JSON que contienen listas de diccionarios en un solo diccionario.
'''

''''
import json

# Leer el archivo con la lista de diccionarios
with open('drug_gene_output.json', 'r', encoding='utf-8') as f:
    lista = json.load(f)
    print('\n\t----------------------La lista de diccionarios tiene:', len(lista) ,'elementos----------------------\n')

# Convertir la lista de diccionarios en un solo diccionario
resultado = {}
for d in lista:
    resultado.update(d)

print('\n\t----------------------La lista de diccionarios tiene:', len(resultado) ,'elementos----------------------\n')
    
# (Opcional) Guardar el resultado en un nuevo archivo
with open('resultado.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

# Imprimir el resultado en pantalla

'''
'''
with open('ATC_farmaco(ENG_dict).json', 'r', encoding='utf-8') as f:
    data1 = json.load(f)
print("Claves de ATC_farmaco(ENG_dict).json:", data1.keys())

with open('drug_gene_output.json', 'r', encoding='utf-8') as f:
    data2 = json.load(f)
print("Claves de drug_gene_output.json:", data2.keys())
'''

with open('train_data/jeso_ATC-drugs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for k,v in data.items():
    match = re.match(r'(.*?)\s*\+\s*(.*?)', v) # Busca los values que tengan una estructura concreta de [String][spaces](+)[spaces][String]
    if match:
        print (f"Clave: '{k}' ----  {v} --- [{', '.join(match.groups())}]")