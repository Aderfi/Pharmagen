import pandas as pd
import json

# --- 1. Carga de Archivos ---

# Rutas a tus archivos de entrada
ruta_csv = 'var_drug_ann_with_ATC.csv'
ruta_json = 'ATC_farmaco(ENG_dict).json'
nombre_columna_farmacos = 'Drug' 
nombre_columna_nueva = 'ATC.1' 

# Cargar el diccionario de traducción desde el archivo JSON
try:
    with open(ruta_json, 'r', encoding='utf-8') as f:
        diccionario_atc = json.load(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo JSON en la ruta: {ruta_json}")
    exit()

# Cargar los datos del CSV en un DataFrame de pandas
try:
    df = pd.read_csv(ruta_csv, sep=';')
except FileNotFoundError:
    print(f"Error: No se encontró el archivo CSV en la ruta: {ruta_csv}")
    exit()

# --- 2. Procesamiento con Bucle 'for' ---

print("Procesando los datos con un bucle for...")

# Creamos una lista vacía para guardar los resultados de cada fila
lista_resultados = []
diccionario_atc = dict(diccionario_atc)
df = pd.DataFrame(df)

df.insert(0, 'ATC.1', None)  # Añadir una columna de índice para referencia

# Iteramos sobre cada fila del DataFrame con df.iterrows()
# 'index' es el número de la fila y 'row' contiene todos los datos de esa fila
for index, row in df.iterrows():
    # Obtenemos el valor de la celda en la columna de fármacos
    celda_farmacos = row['Drug']
    celda_nueva = row['ATC.1']
    resultado_final_fila = "" # Variable para guardar el resultado de esta fila
        # Si detectamos una coma, significa que hay múltiples fármacos
    if ',' in celda_farmacos:
            # 1. Separamos los fármacos por la coma y limpiamos espacios
            lista_farmacos = [f.strip() for f in str(celda_farmacos).split(',')]
            
            # 2. Creamos una lista para las traducciones individuales
            traducciones_individuales = []
            
            # 3. Recorremos la lista de fármacos para traducir cada uno
            for farmaco in lista_farmacos:
                # Usamos .get() para traducir de forma segura
                trad = diccionario_atc.get(farmaco)
                traducciones_individuales.append(trad)

            df.at[df.index[0],'ATC_1'] = ', '.join(traducciones_individuales)

    elif '/' in celda_farmacos:
            # 1. Separamos los fármacos por la coma y limpiamos espacios
            lista_farmacos = [f.strip() for f in str(celda_farmacos).split('/')]
            
            # 2. Creamos una lista para las traducciones individuales
            traducciones_individuales = []
            
            # 3. Recorremos la lista de fármacos para traducir cada uno
            for farmaco in lista_farmacos:
                # Usamos .get() para traducir de forma segura
                trad = diccionario_atc.get(farmaco)
                traducciones_individuales.append(trad)

            df.at[row.index, 'ATC.1'] = ', '.join(traducciones_individuales)

    # Si no hay coma, es un único fármaco
    else:
        traducciones_individuales = []

        farmaco_individual = (str(celda_farmacos).strip())  # Limpiamos espacios
        farmaco = diccionario_atc.get(farmaco_individual)
        farmaco = str(farmaco).strip() if farmaco is not None else ""

        df.at[row.index, 'ATC.1'] = df['ATC.1'].replace('nan' or '', farmaco)
    # Rellenamos NaN con cadenas vacías

    # Añadimos el resultado de la fila (ya sea traducido o vacío) a nuestra lista principal
   # lista_resultados.append(resultado_final_fila)

# --- 3. Asignación de Resultados y Salida ---

# Asignamos la lista completa de resultados a la nueva columna del DataFrame.
# La lista tiene el mismo orden y tamaño que las filas del DataFrame.

print("\n" + "="*40 + "\n")
print("DataFrame con la Columna Traducida:")
print(df)

# Guardamos el resultado en un nuevo archivo CSV
ruta_salida_csv = 'resultados_traducidos_bucle.csv'
df.to_csv(ruta_salida_csv, sep=';', index=False, encoding='utf-8-')

print(f"\n✅ ¡Proceso completado! Resultados guardados en: {ruta_salida_csv}")