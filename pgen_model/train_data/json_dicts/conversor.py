import pandas as pd
import json
import numpy as np # Es buena práctica importarlo

# 1. Cargar los datos
df = pd.read_csv('train.csv', sep=';')
with open('dict_therapeutic_outcome_gemini.json', 'r', encoding='utf-8') as f:
    dict_therapeutic_outcome = json.load(f)

# 2. [IMPORTANTE] Estandarizar las llaves del diccionario
# Las llaves en tu JSON tienen una mezcla de mayúsculas, minúsculas y 'nan'/'NAN'.
# Para un mapeo fiable, convertimos todas las llaves del diccionario a mayúsculas.
dict_therapeutic_outcome = {k.upper(): v for k, v in dict_therapeutic_outcome.items()}

# 3. Preparar las columnas del DataFrame
# Hacemos lo mismo con las columnas del DataFrame:
# - Rellenamos los valores nulos (NaN) con la cadena 'NAN'.
# - Convertimos todo a string.
# - Convertimos todo a mayúsculas.
cols_to_map = ['Outcome_category', 'Effect_direction', 'Effect_category']
# 4. Crear la llave de la triada
# Concatenamos las tres columnas para que coincidan con el formato del diccionario
df['triad_key'] = df['Outcome_category'] + ';' + \
                    df['Effect_direction'] + ';' + \
                    df['Effect_category']

# 5. Mapear los valores a la nueva columna
# Usamos .map() que es la función ideal para esto.
# Si no encuentra una llave en el diccionario, asignará NaN (Not a Number)
df['Therapeutic_Outcome'] = df['triad_key'].map(dict_therapeutic_outcome)

# 6. (Opcional) Eliminar la columna de llave temporal
df = df.drop(columns=['triad_key'])

# 7. (Opcional) Ver el resultado
df.to_csv('train_therapeutic_outcome.csv', sep=';', index=False)