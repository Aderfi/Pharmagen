import json

# Rutas de entrada y salida
json_path = "atc_tree.json"
output_path = "farmacos.json"

# Carga el árbol ATC (diccionario en la raíz)
with open(json_path, "r", encoding="utf8") as f:
    atc_tree = json.load(f)

def extraer_farmacos(nodo, resultado):
    if isinstance(nodo, dict):
        cod = nodo.get("codigo", "")
        if not nodo.get("children") and len(cod) >= 6:
            resultado.append({
                "codigo": cod,
                "descripcion": nodo.get("descripcion", "")
            })
        for hijo in nodo.get("children", []):
            extraer_farmacos(hijo, resultado)
    elif isinstance(nodo, list):
        for item in nodo:
            extraer_farmacos(item, resultado)
    elif isinstance(nodo, dict):  # Por si acaso
        for v in nodo.values():
            extraer_farmacos(v, resultado)

farmacos = []
# Recorre todos los valores del diccionario raíz
for clave in atc_tree:
    extraer_farmacos(atc_tree[clave], farmacos)

with open(output_path, "w", encoding="utf8") as f:
    json.dump(farmacos, f, ensure_ascii=False, indent=2)

print(f"Total fármacos encontrados: {len(farmacos)}")
print(f"Lista guardada en {output_path}")