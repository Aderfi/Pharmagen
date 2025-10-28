import pandas as pd
import re
import json
import numpy as np
import itertools
import sys
df = pd.read_csv("train_base.csv", sep=";", index_col=False)
'''
#Drug;Gene;Allele;Genotype;Outcome;Variation;Effect_subcat;Population_Affected;Entity_Affected;Therapeutic_Outcome


cols = ['Drug', 'Gene', 'Allele', 'Genotype', 
        'Outcome', 'Variation', 'Effect', 'Effect_type', 'Effect_name', 'Entity',
        'Entity_type','Entity_name', 'Population_Affected', 'Therapeutic_Outcome']

df.rename(
          columns={'Effect_subcat': 'Effect'
                   , 'Entity_Affected': 'Entity'
                   , 'Population_Affected': 'Pop_aff'},
		  inplace=True)

df = df.reindex(cols, axis=1)

#cols = ['Genotype', 'Outcome', 'Variation', 'Effect', 'Effect_name', 'Entity']

#df['Name'] = df['Effect'].astype(str) + ', ' + df['Entity'].astype(str)

Effects_list = ['SIDE_EFFECT', 'DISEASE', 'OTHER', 'EFFICACY', 'PK']

#df = df[df['Effect'].isin(Effects_list)]

def arreglar_effects(x):
    # 1. Comprobar NaN o None ANTES de convertir a string
    if pd.isna(x):
        return np.nan
    
    # 2. Convertir a string y limpiar espacios en blanco
    x = str(x).strip()
    
    # 3. Comprobar si es una cadena vacía
    if x == '':
        return np.nan

    # 4. Procesar la cadena
    # Dividimos por comas (si no hay comas, nos da una lista de 1 elemento)
    parts = x.split(',')
    
    valid_parts = []
    for part in parts:
        # Limpiamos cada parte: quitamos espacios y lo de después de ":"
        cleaned_part = part.strip().split(':')[0].strip()
        
        # Añadimos solo si es válido y no es una cadena vacía
        if cleaned_part in Effects_list and cleaned_part != '':
            valid_parts.append(cleaned_part)

    # 5. Devolver el resultado
    if not valid_parts: # Si la lista está vacía
        return np.nan
    else:
        # Unimos las partes válidas con ", "
        return ', '.join(set(valid_parts))


#df['Effect'] = df['Effect'].apply(arreglar_effects)

################################################################


import pandas as pd
import numpy as np

def procesar_entidad(entity_str):
    """
    Procesa una cadena de entidad compleja y la divide en tipos y nombres.
    
    Entrada: "SIDE_EFFECT:HEART_ATTACK, OTHER:TOXIC_ADVERSE, EFFICACY:RESPONSE"
    Salida: pd.Series(["SIDE_EFFECT, OTHER, EFFICACY", "HEART_ATTACK, TOXIC_ADVERSE, RESPONSE"])
    """
    
    # 1. Manejar valores nulos o vacíos primero
    if pd.isna(entity_str) or not str(entity_str).strip():
        return pd.Series([np.nan, np.nan])

    entity_str = str(entity_str)
    
    # Listas para guardar las partes
    lista_tipos = []
    lista_nombres = []
    
    # 2. Dividir la cadena en pares (ej. "SIDE_EFFECT:HEART_ATTACK")
    #    Usamos .strip() para limpiar espacios en blanco
    pares = [par.strip() for par in entity_str.split(',')]
    
    # 3. Procesar cada par
    for par in pares:
        # Usamos partition() en lugar de split(). Es más seguro porque
        # siempre devuelve 3 partes: (antes_del_separador, separador, después_del_separador)
        # Si no encuentra el ':', devuelve (cadena, "", "")
        
        tipo, separador, nombre = par.partition(':')
        
        # Si el separador (':') se encontró:
        if separador:
            lista_tipos.append(tipo.strip())
            lista_nombres.append(nombre.strip())
            
    # 4. Si no se encontró ningún par válido, devolver NaN
    if not lista_tipos:
        return pd.Series([np.nan, np.nan])

    # 5. Unir las listas y devolverlas como una Serie de Pandas
    #    Pandas expandirá esta Serie en las dos columnas de destino.
    return pd.Series([
        ", ".join(lista_tipos),
        ", ".join(lista_nombres)
    ])

# --- Aplicación en el DataFrame ---

# Asumimos que 'df' es tu DataFrame y 'Entity' es la columna
df[['Entity_type', 'Entity_name']] = df['Entity'].apply(procesar_entidad)
df[['Effect_type', 'Effect_name']] = df['Effect'].apply(procesar_entidad)

print(df[['Effect', 'Effect_type', 'Effect_name']].sample(20))

df.to_csv("BACKUPS/backup_train.csv", sep=";", index=False)

df.drop(columns=['Entity', 'Effect'], inplace=True)

df.to_csv("train_base.csv", sep=";", index=False)
'''


# ['Drug', 'Gene', 'Allele', 'Genotype', 'Outcome', 'Variation', 'Effect_type', 'Effect_name', 'Entity_type', 'Entity_name', 'Population_Affected', 'Therapeutic_Outcome'],

df['Type'] = df['Effect_type'].astype(str) + ', ' + df['Entity_type'].astype(str)
df['Type'] = df['Type'].str.replace('nan,', '', regex=False).str.strip()
df['Type'] = df['Type'].str.replace(', nan', '', regex=False).str.strip()
df['Type'] = df['Type'].replace('nan', '', regex=False)


df['Name'] = df['Effect_name'].astype(str) + ', ' + df['Entity_name'].astype(str)
df['Name'] = df['Name'].str.replace('nan,', '', regex=False).str.strip()
df['Name'] = df['Name'].str.replace(', nan', '', regex=False).str.strip()
df['Name'] = df['Name'].replace('nan', '', regex=False)

df.drop(columns=['Effect_type', 'Effect_name', 'Entity_type', 'Entity_name'], inplace=True)

df = df.reindex(columns=['Drug', 'Gene', 'Allele', 'Genotype', 'Outcome', 'Variation',
				   'Type', 'Name', 'Therapeutic_Outcome'])


#df['Test'] = df.join(df[['Effect_type', 'Entity_type']].astype(str).agg(', '.join, axis=1))

with open('mapeo_geno_alleles_dict.json', 'r') as f:
	mapping_dict = json.load(f)

df['Genotype'] = df['Genotype'].map(mapping_dict)

df.to_csv("train_merged.csv", sep=";", index=False)
print(df['Genotype'].sample(50))
