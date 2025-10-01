#   -*- coding: utf-8 -*-
# I/O functions for Pharmagen_PModel
import json

# Función para para tokenizar los inputs que van a ser procesador por el modelo predictivo
# de machine learning para la predicción de eficacia y toxicidad de fármacos en pacientes
# de acuerdo a su genotipo. 

def tokenizar_inputs(input_data, output_file):
    """Tokeniza los datos de entrada y los guarda en un archivo JSON."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(input_data, f, ensure_ascii=False, indent=4)
        print(f"Datos tokenizados guardados en {output_file}")
    except Exception as e:
        print(f"Error al guardar los datos tokenizados: {e}")
    return output_file