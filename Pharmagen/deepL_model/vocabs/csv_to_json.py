import pandas as pd
import json
import re
from unidecode import unidecode

# Ruta a tu archivo CSV y salida JSON
csv_path = "Tabla_ATC.csv"
json_path = "atc_tree.json"

# Lee el archivo CSV 
df = pd.read_csv(csv_path, encoding="latin1", sep=";")

# Limpia y estandariza nombres de columnas
df.columns = [unidecode(col.strip().replace(" ", "_")).upper() for col in df.columns]
df = df[df.columns[:3]]  # Mantén solo las primeras 3 columnas relevantes
print("Nombres de columnas después de limpiar:", df.columns.tolist())


# Normalizacion de los datos
df['COD_ATC'] = df['COD_ATC'].astype(str).str.strip()
df['DESCRIPCION'] = df['DESCRIPCION'].astype(str).str.strip()
df['NIVEL_ATC'] = df['NIVEL_ATC'].astype(int)

# Diccionarios para nodos
nodos = {}
atc_tree = {}

for _, row in df.iterrows():
    nivel = row["NIVEL_ATC"]
    cod = row["COD_ATC"]
    desc = row["DESCRIPCION"]

    nodo = {"codigo": cod, "descripcion": desc, "children": []}

    nodos[cod] = nodo

    if nivel == 1:
        atc_tree[cod] = nodo
    else:
        padre = None
        for i in range(len(cod) - 1, 0, -1):
            prefijo = cod[:i]
            if prefijo in nodos:
                padre = nodos[prefijo]
                break
        if padre:
            padre["children"].append(nodo)
        else:
            atc_tree[cod] = nodo
# Export a JSON
with open(json_path, "w", encoding="utf8") as f:
    json.dump(atc_tree, f, ensure_ascii=False, indent=2)

print(f"Exportado a {json_path}")