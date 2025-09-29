#   -*- coding: utf-8 -*-
# I/O functions for Pharmagen_PModel
import json

# Función para para tokenizar los inputs que van a ser procesador por el modelo predictivo
# de machine learning para la predicción de eficacia y toxicidad de fármacos en pacientes
# de acuerdo a su genotipo. 

def tokenizar_inputs(input_data, output_file):
    # input_data: datos de entrada en formato bruto (string, lista, diccionario, etc.)
    # output_file: ruta al archivo donde se guardarán los datos tokenizados (JSON)
    
    try:
        # Aquí se implementaría la lógica de tokenización específica según el formato esperado
        # Por ejemplo, si el input es un string con datos separados por comas:
        if isinstance(input_data, str):
            tokens = input_data.split(',')
        elif isinstance(input_data, list):
            tokens = input_data
        elif isinstance(input_data, dict):
            tokens = [f"{k}:{v}" for k, v in input_data.items()]
        else:
            raise ValueError("Formato de input no soportado")
        
        # Guardar los tokens en un archivo JSON
        with open(output_file, 'w') as f:
            json.dump(tokens, f)
        
        print(f"Datos tokenizados y guardados en {output_file}")
        return tokens
    
    except Exception as e:
        print(f"Error al tokenizar los datos: {e}")
        return None

# Función para para "destokenizar" los outputs ya procesador por el modelo predictivo
# de machine learning con predicciones de eficacia y toxicidad de fármacos en pacientes
# de acuerdo a su genotipo.

def destokenizar_outputs(tokenized_data, output_file):
    # tokenized_data: datos tokenizados (lista de strings)
    # output_file: ruta al archivo donde se guardarán los datos destokenizados (JSON)
    
    try:
        # Aquí se implementaría la lógica de destokenización específica según el formato esperado
        # Por ejemplo, si los tokens son strings en formato "clave:valor":
        if isinstance(tokenized_data, list):
            destokens = {}
            for token in tokenized_data:
                if ':' in token:
                    k, v = token.split(':', 1)
                    destokens[k] = v
                else:
                    raise ValueError("Token no tiene el formato esperado 'clave:valor'")
        else:
            raise ValueError("Formato de tokenized_data no soportado")
        
        # Guardar los datos destokenizados en un archivo JSON
        with open(output_file, 'w') as f:
            json.dump(destokens, f)
        
        print(f"Datos destokenizados y guardados en {output_file}")
        return destokens
    
    except Exception as e:
        print(f"Error al destokenizar los datos: {e}")
        return None