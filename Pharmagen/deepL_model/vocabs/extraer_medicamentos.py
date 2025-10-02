import json
import pandas as pd

# Rutas de entrada y salida
json_path = "atc_tree.json"
output_path = "farmacos.json"

# Carga el árbol ATC
with open(json_path, "r", encoding="utf8") as f:
    atc_tree = json.loads(f.read(), object_pairs_hook=dict)

i = 0

for k in atc_tree.keys():
    for v in atc_tree[k].values():
        print(k, v)
        i += 1
        if i > 1:
            break
        

def extraer_farmacos(nodo, resultado):
    if isinstance(nodo, dict):
        cod = nodo.get("codigo", "")
        
        # Es hoja (sin hijos) y código >= 6 caracteres
        if len(cod) >= 6:
            resultado.append({
                "codigo": cod,
                "descripcion": nodo.get("descripcion", "")
            })
        for hijo in nodo.get("children", []):
            extraer_farmacos(hijo, resultado)
    elif isinstance(nodo, list):
        for item in nodo:
            extraer_farmacos(item, resultado)

farmacos = []
extraer_farmacos(atc_tree, farmacos)

with open(output_path, "w", encoding="utf8") as f:
    json.dump(farmacos, f, ensure_ascii=False, indent=2)

print(f"Total fármacos encontrados: {len(farmacos)}")
print(f"Lista guardada en {output_path}")

