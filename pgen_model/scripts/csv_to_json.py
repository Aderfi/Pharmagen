import csv
import json
import pandas as pd 


def convert_csv_to_json():
    # Solicitar al usuario el numero de archivos
    
    print
    
    csv_file = input("Ingrese la ruta del archivo CSV (con columnas 'nombre' y 'codigo_atc'): ")
    json_file = input("Ingrese la ruta donde desea guardar el archivo JSON: ")

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