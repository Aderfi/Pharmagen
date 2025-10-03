import csv
import json
import pandas as pd

csv_path = 'eng_atc_tree.csv'
json_path = 'eng_atc_tree_from_csv.json'

# Leemos el CSV usando pandas
df = pd.read_csv(csv_path)

# Creamos la lista de diccionarios en formato [{codigo: nombre}, ...]
result = []
for idx, row in df.iterrows():
    nombre = str(row['nombre']).strip()
    codigo = str(row['codigo_atc']).strip()
    if codigo and nombre:
        result.append({codigo: nombre})

# Guardamos como JSON
with open(json_path, 'w', encoding='utf8') as f_json:
    json.dump(result, f_json, indent=2, ensure_ascii=False)