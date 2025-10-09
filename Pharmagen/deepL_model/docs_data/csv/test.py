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


with open('train_data/jeso_ATC-drugs.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for k,v in data.items():
    match = re.match(r'(.*?)\s*\+\s*(.*?)', v) # Busca los values que tengan una estructura concreta de [String][spaces](+)[spaces][String]
    if match:
        print (f"Clave: '{k}' ----  {v} --- [{', '.join(match.groups())}]")
'''

	#Nivel ATC;COD_ATC;Descripcion;;;;;;
	#1;A;APARATO DIGESTIVO Y METABOLISMO;;;;;;
	#2;A01;PREPARADOS ESTOMATOLOGICOS;;;;;;
	#3;A01A;PREPARADOS ESTOMATOLOGICOS;;;;;;
	#4;A01AA;PREPARADOS PREVENTIVOS DE LA CARIES;;;;;;
	#5;A01AA01;FLUORURO SODICO;;;;;;

'''atc_df_dict = {}
jerarquia_dict = {}


with open('Tabla_ATC.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=';')
    next(reader)  # Saltar la fila de encabezado si existe
    for row in reader:
        jerarquia = row[0]
        cod_atc = row[1]
        medicamento = row[2]
        atc_df_dict[cod_atc] = medicamento
        
        
      
        
            


with open('Jerarquia_ATC.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter=';')
    writer.writerow(['ATC', 'Medicamento', 'Anatomico', 'Terapeutico', 'Farmacologico', 'Quimico', 'Medicamento'])
    for cod_atc, descripcion in atc_df_dict.items():
        writer.writerow([cod_atc, descripcion])'''
''''''

'''
import pandas as pd

# --- Paso 1: Cargar los datos ---
# Lee el archivo CSV. Asegúrate de que el delimitador sea el correcto (';').
# No es necesario abrir los archivos manualmente con 'with open', Pandas lo maneja por ti.
try:
    #ATC;Medicamento;Anatomico;Terapeutico;Farmacologico;Quimico;Medicamento

    df_in = pd.read_csv('Jerarquia_ATC.csv', delimiter=';', usecols=['ATC', 'Medicamento'], dtype='string', )
except FileNotFoundError:
    print("Error: El archivo 'Jerarquia_ATC.csv' no se encontró.")
    # Puedes crear un archivo de ejemplo si no existe para probar el código
    # pd.DataFrame({'ATC': ['A01AA01', 'N02BE01', 'C09AA02'], 'Medicamento': ['Fluoruro de sodio', 'Paracetamol', 'Enalapril']}).to_csv('Jerarquia_ATC.csv', sep=';', index=False)
    # df = pd.read_csv('Jerarquia_ATC.csv', delimiter=';', usecols=['ATC', 'Medicamento'], dtype='string')
    exit()
    
# --- Paso 2: Crear las columnas jerárquicas usando slicing de strings ---
# Las operaciones de string en Pandas (.str) son vectorizadas, lo que significa
# que se aplican a toda la columna a la vez. Es mucho más rápido que un bucle for.

# Eliminamos filas donde el código ATC pueda ser nulo o inválido para evitar errores.
#df = df[df['ATC'].str.len()]

df = pd.DataFrame(columns=['ATC', 'Medicamento', 'Anatomico', 'Terapeutico', 'Farmacologico', 'Quimico', 'Medicamento'])
df['ATC'] = df_in['ATC']
df['Medicamento'] = df_in['Medicamento']

for index, row in df.iterrows():
    row = pd.Series(row)
    row = row.astype(str)
    # Creamos las nuevas columnas extrayendo partes del código ATC
    if len(row['ATC']) == 1:
        df['Anatomico'] = df['ATC'].str[0]       # 1er nivel: 1 carácter (ej. A)
        df['Terapeutico'] = 'kuk'      # 2º nivel: 3 caracteres (ej. A01)
        df['Farmacologico'] = 'kuk'  # 3er nivel: 4 caracteres (ej. A01A)
        df['Quimico'] = 'kuk'      # 4º nivel: 5 caracteres (ej. A01AA)
    # El 5º nivel (Medicamento específico) es el código completo, que ya tenemos en 'ATC'.

    # Creamos las nuevas columnas extrayendo partes del código ATC
    elif len(row['ATC']) == 2:
        df['Anatomico'] = df['ATC'].str[0]       # 1er nivel: 1 carácter (ej. A)
        df['Terapeutico'] = df['ATC'].str[0:3]     # 2º nivel: 3 caracteres (ej. A01)
        df['Farmacologico'] = 'kuk'  # 3er nivel: 4 caracteres (ej. A01A)
        df['Quimico'] = 'kuk'       # 4º nivel: 5 caracteres (ej. A01AA)
    # El 5º nivel (Medicamento específico) es el código completo, que ya tenemos en 'ATC'.

    # Creamos las nuevas columnas extrayendo partes del código ATC
    elif len(row['ATC']) == 3:
        df['Anatomico'] = df['ATC'].str[0]       # 1er nivel: 1 carácter (ej. A)
        df['Terapeutico'] = df['ATC'].str[0:3]     # 2º nivel: 3 caracteres (ej. A01)
        df['Farmacologico'] = df['ATC'].str[0:4] # 3er nivel: 4 caracteres (ej. A01A)
        df['Quimico'] = 'kuk'      # 4º nivel: 5 caracteres (ej. A01AA)
    # El 5º nivel (Medicamento específico) es el código completo, que ya tenemos en 'ATC'.

    # Creamos las nuevas columnas extrayendo partes del código ATC
    elif len(row['ATC']) == 4:
        df['Anatomico'] = df['ATC'].str[0]       # 1er nivel: 1 carácter (ej. A)
        df['Terapeutico'] = df['ATC'].str[0:3]     # 2º nivel: 3 caracteres (ej. A01)
        df['Farmacologico'] = df['ATC'].str[0:4] # 3er nivel: 4 caracteres (ej. A01A)
        df['Quimico'] = df['ATC'].str[0:5]      # 4º nivel: 5 caracteres (ej. A01AA)
    # El 5º nivel (Medicamento específico) es el código completo, que ya tenemos en 'ATC'.
    else:
        continue


print(df.head(10),) 


#24;A01AC01;TRIAMCINOLONA;A;A01;A01A;A01AC;TRIAMCINOLONA
for row in range(len(df)):
    row = pd.Series(df.iloc[row])
    row = row.astype(str)
    
    if len(row[0]) == 1:
        df.loc[row.name, ['Terapeutico', 'Farmacologico', 'Quimico']] = '', '', ''  # 3
          
    elif len(row[0]) == 3:
        df.loc[row.name, ['Farmacologico', 'Quimico']] = '', ''  # 3

    elif len(row[0]) == 4:
        df.loc[row.name, ['Quimico']] = ''  # 4


print(df.head(10)) 



# --- Paso 3: Reordenar las columnas para que coincidan con tu DataFrame original ---
# Creamos una copia para evitar 'SettingWithCopyWarning' y aseguramos el orden.
'''

import json
import pandas as pd
import csv
'''
# --- Paso 4: Guardar el resultado en un nuevo archivo CSV ---
#df.to_csv('Jerarquia_ATC_Pandas.csv', columns=['ATC', 'Medicamento', 'Anatomico', 'Terapeutico', 'Farmacologico', 'Quimico'], sep=';', encoding='utf-8', index=False)



with open('ATC_completo_ESP.csv', 'r', newline='', encoding='utf-8') as f:
    df = pd.read_csv(f, delimiter=';', usecols=['ATC', 'Medicamento'], dtype='string')

print(df.head(10))
json_dict = df
with open('ATC_completo_ESP.json', 'w', encoding='utf-8') as f:
    json_dict = pd.DataFrame.to_dict(json_dict, orient='records')
    json.dump(json_dict, f, ensure_ascii=False, indent=2)
print("¡Proceso completado! El archivo 'ATC_completo_ESP.json' ha sido creado.")

with open('list_drugs.txt', 'w', encoding='utf-8') as f:
    drugs_list = [list(v for k,v in entry.items())[1] for entry in json_dict]
    f.write('\n'.join(str(drug).upper() for drug in drugs_list))
'''
'''
atc_df_dict_list = []

atc_df_dict = dict(atc_df_dict)

for i in range(len(atc_df_dict)):
   for k,v in atc_df_dict.items():
       if v == '':
           atc_df_dict_list.append([k,v])

for index in atc_df_dict_list:
    atc_df_dict.pop(index[0], index[1])

print(atc_df_dict)
'''

with open ('ATC_completo_ES-EN.csv', 'r', encoding='utf-8') as f, open('ATC_drug_dict_ESP.json', 'w', encoding='utf-8') as f2, open('ATC_drug_dict_ENG.json', 'w', encoding='utf-8') as f3:
    df = pd.read_csv(f, delimiter=';', dtype='string', usecols=['ATC', 'Medicamento', 'Drug', 'Anatomico', 'Terapeutico', 'Farmacologico', 'Quimico'])
    
    columnas = ['ATC', 'Medicamento', 'Drug',  'Anatomico', 'Terapeutico', 'Farmacologico', 'Quimico']
    
    json_out_cols_ES = ['ATC', 'Medicamento']
    json_out_cols_EN = ['ATC', 'Drug']

    dict_es = {}
    dict_en = {}

    
    for index, row in df.iterrows():
        row = pd.Series(row)
        row = row.astype(str)

        cod_atc = row['ATC'].strip().upper()
        medicamento_esp = row['Medicamento'].strip().upper()
        medicamento_eng = row['Drug'].strip().upper()

        dict_es[cod_atc] = medicamento_esp
        dict_en[cod_atc] = medicamento_eng
    
    json.dump(dict_es, f2, ensure_ascii=False, indent=2)
    json.dump(dict_en, f3, ensure_ascii=False, indent=2)    

