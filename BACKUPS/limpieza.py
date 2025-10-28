'''
import pandas as pd
import sys
import re

# --- CONFIGURACIÓN ---
INPUT_CSV_PATH = 'train_base_therapeutic.csv' # <- CAMBIA ESTO
OUTPUT_CSV_PATH = 'train_base_therapeutic_limpio.csv' # <- CAMBIA ESTO
COLUMN_TO_CLEAN = 'Drugs'
# --- FIN DE LA CONFIGURACIÓN ---

def clean_drug_cell(cell_content):
    """
    Limpia una celda de la columna 'Drugs'.
    Respeta las comas como separadores de lista.
    Ej: "A B, C D" -> "A_B, C_D"
    """
    # 1. Comprobar si la celda es un string
    if not isinstance(cell_content, str):
        # Si es NaN, None, o un número, devolverlo tal cual
        return cell_content

    # 2. Dividir el string en una lista, usando la coma como separador
    drug_list = cell_content.split(',')
    
    # 3. Procesar cada elemento de la lista
    cleaned_list = []
    for drug_name in drug_list:
        # a. Quitar espacios en blanco al principio y al final
        cleaned_drug = drug_name.strip()
        
        # b. Reemplazar espacios internos con guiones bajos
        # Usamos re.sub para reemplazar cualquier tipo de espacio (\s)
        replaced_drug = re.sub(r'\s+', '_', cleaned_drug)
        
        cleaned_list.append(replaced_drug)
    
    # 4. Volver a unir la lista en un solo string, separado por ", "
    return ', '.join(cleaned_list)


# --- Función Principal ---
def main():
    print(f"Cargando CSV de entrada: {INPUT_CSV_PATH}")
    try:
        # Usamos dtype=str para asegurarnos de que todo se lea como texto
        df = pd.read_csv(INPUT_CSV_PATH, sep=';', dtype=str)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {INPUT_CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if COLUMN_TO_CLEAN not in df.columns:
        print(f"Error: La columna '{COLUMN_TO_CLEAN}' no se encuentra en el CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Procesando la columna '{COLUMN_TO_CLEAN}'...")
    
    # Aplicar la función de limpieza a cada celda de la columna
    df[COLUMN_TO_CLEAN] = df[COLUMN_TO_CLEAN].apply(clean_drug_cell)

    print(f"Guardando CSV limpio en: {OUTPUT_CSV_PATH}")
    try:
        df.to_csv(OUTPUT_CSV_PATH, sep=';', index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}", file=sys.stderr)
        sys.exit(1)

    print("¡Proceso completado con éxito!")


if __name__ == "__main__":
    main()
    
# Variant_id;ATC;Drugs;Gene;Alleles;Genotype;Outcome_category;Effect_direction;Effect_category;Entity;Entity_name;Affected_Pop;Sentence;Notes
import pandas as pd
import itertools
import json
import json
import numpy as np
import sys
import re

df = pd.read_csv('train_base.csv', sep=';', usecols=['Drugs', 'Gene', 'Alleles', 'Genotype', 'Outcome', 'Effect_direction', 'Effect', 'Effect_subcat', 'Population_Affected', 'Entity_Affected'])

# import dict
with open('dict_therapeutic_outcome.json') as f:
    ther_out_dict = json.load(f)
### CONFIGURACIÓN ###

INPUT_CSV_PATH = 'train_base.csv' # <- CAMBIA ESTO
OUTPUT_CSV_PATH = 'train_base_therapeutic.csv' # <- CAMBIA ESTO

# ---- Mapeo de Therapeutic Outcome ----
JSON_FILE_PATH = 'dict_therapeutic_outcome.json'

KEY_COLUMNS = ['Outcome', 'Effect_direction', 'Effect']
NEW_COLUMN_NAME = 'Therapeutic_Outcome'

# --- Limpieza de columnas

COLUMN_TO_CLEAN = 'Drugs'

# --- FIN DE LA CONFIGURACIÓN ---





def load_mapping_dict(json_path):
    """
    Carga el diccionario JSON y normaliza sus claves a mayúsculas
    para un mapeo robusto e insensible a mayúsculas/minúsculas.
    """
    print(f"Cargando el diccionario de mapeo desde: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            mapping_dict = json.load(f)
        
        # Normalizar claves: todo a mayúsculas para evitar errores
        # 'DOSAGE;nan;DOSE' -> 'DOSAGE;NAN;DOSE'
        normalized_dict = {key.upper(): value for key, value in mapping_dict.items()}
        print(f"Diccionario cargado y normalizado ({len(normalized_dict)} claves).")
        return normalized_dict
        
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo JSON en {JSON_FILE_PATH}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: El archivo {JSON_FILE_PATH} no es un JSON válido.", file=sys.stderr)
        sys.exit(1)

def map_csv(input_path, output_path, mapping_dict):
    """
    Lee el CSV de entrada, aplica el mapeo y guarda el CSV de salida.
    """
    print(f"Cargando el archivo CSV de entrada: {input_path}")
    try:
        # Asumiendo separador por punto y coma, como en tus datos de muestra
        # dtype=str evita que pandas interprete 'nan' como un valor nulo de tipo float
        df = pd.read_csv(input_path, sep=';', dtype=str, usecols=['Drugs', 'Gene', 'Alleles', 'Genotype', 'Outcome', 'Effect_direction', 'Effect', 'Effect_subcat', 'Population_Affected', 'Entity_Affected'])
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {input_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    print("Normalizando columnas y creando clave de mapeo...")
    
    # 1. Crear una clave de mapeo temporal
    # Usamos .copy() para evitar advertencias de 'SettingWithCopyWarning'
    df_keys = df[KEY_COLUMNS].copy()
    
    # 2. Normalizar las columnas del DataFrame (igual que el diccionario)
    for col in KEY_COLUMNS:
        # Rellenar nulos (None, np.nan, etc.) con el string 'NAN'
        # y convertir todo a mayúsculas
        df_keys[col] = df_keys[col].fillna('NAN').astype(str).str.upper()
        # Asegurarse de que los strings 'nan' también se conviertan
        df_keys[col] = df_keys[col].replace('NAN', 'NAN', regex=False)

    # 3. Crear la clave combinada (ej: "DOSAGE;NAN;DOSE")
    df['_outcome_first'] = df_keys[KEY_COLUMNS[0]].str.split(',').str[0].str.strip()
    df['_map_key'] = df['_outcome_first'] + ';' + \
                     df_keys[KEY_COLUMNS[1]] + ';' + \
                     df_keys[KEY_COLUMNS[2]]

    print("Aplicando el mapeo...")
    # 4. Aplicar el mapeo para crear la nueva columna
    df[NEW_COLUMN_NAME] = df['_map_key'].map(mapping_dict)

    # --- Reporte y Limpieza ---
    nan_count = df[NEW_COLUMN_NAME].isna().sum()
    total_rows = len(df)
    
    print(f"Mapeo completado. {total_rows - nan_count} de {total_rows} filas fueron mapeadas.")
    if nan_count > 0:
        print(f"ADVERTENCIA: {nan_count} filas no encontraron una clave en el diccionario. Se les asignará 'NaN'.")
        # Descomenta la línea de abajo si quieres ver las 10 primeras claves que fallaron
        # print(f"  Algunas claves que fallaron: {df[df[NEW_COLUMN_NAME].isna()]['_map_key'].unique()[:10]}")
        missing_keys = df[df[NEW_COLUMN_NAME].isna()]['_map_key'].unique()
        for key in missing_keys:
            print(f"    - {key}")

    # 5. Eliminar la columna de clave temporal
    df = df.drop(columns=['_map_key'])
    df = df.drop(columns=['_outcome_first'])

    print(f"Guardando el archivo CSV actualizado en: {output_path}")
    try:
        # Guardar el DataFrame resultante
        # 'utf-8-sig' ayuda a Excel a abrir correctamente los caracteres especiales
        df.to_csv(output_path, sep=';', index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}", file=sys.stderr)
        sys.exit(1)
        

    print("¡Proceso finalizado con éxito!")

def clean_drug_cell(cell_content):
    """
    Limpia una celda de la columna 'Drugs'.
    Respeta las comas como separadores de lista.
    Ej: "A B, C D" -> "A_B, C_D"
    """
    # 1. Comprobar si la celda es un string
    if not isinstance(cell_content, str):
        # Si es NaN, None, o un número, devolverlo tal cual
        return cell_content

    # 2. Dividir el string en una lista, usando la coma como separador
    drug_list = cell_content.split(',')
    
    # 3. Procesar cada elemento de la lista
    cleaned_list = []
    for drug_name in drug_list:
        # a. Quitar espacios en blanco al principio y al final
        cleaned_drug = drug_name.strip()
        
        # b. Reemplazar espacios internos con guiones bajos
        # Usamos re.sub para reemplazar cualquier tipo de espacio (\s)
        replaced_drug = re.sub(r'\s+', '_', cleaned_drug)
        
        cleaned_list.append(replaced_drug)
    
    # 4. Volver a unir la lista en un solo string, separado por ", "
    return ', '.join(cleaned_list)


# --- Función Principal ---
def limpieza_columna():
    print(f"Cargando CSV de entrada: {OUTPUT_CSV_PATH}")
    try:
        # Usamos dtype=str para asegurarnos de que todo se lea como texto
        df = pd.read_csv(OUTPUT_CSV_PATH, sep=';', dtype=str)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo CSV en {OUTPUT_CSV_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer el CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if COLUMN_TO_CLEAN not in df.columns:
        print(f"Error: La columna '{COLUMN_TO_CLEAN}' no se encuentra en el CSV.", file=sys.stderr)
        sys.exit(1)

    print(f"Procesando la columna '{COLUMN_TO_CLEAN}'...")
    
    # Aplicar la función de limpieza a cada celda de la columna
    df[COLUMN_TO_CLEAN] = df[COLUMN_TO_CLEAN].apply(clean_drug_cell)

    print(f"Guardando CSV limpio en: {OUTPUT_CSV_PATH}")
    try:
        df.to_csv(OUTPUT_CSV_PATH, sep=';', index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"Error al guardar el archivo de salida: {e}", file=sys.stderr)
        sys.exit(1)

    print("¡Proceso completado con éxito!")

# --- Ejecutar el script ---
if __name__ == "__main__":
    normalized_dict = load_mapping_dict(JSON_FILE_PATH)
    map_csv(INPUT_CSV_PATH, OUTPUT_CSV_PATH, normalized_dict)
    limpieza_columna()
    
'''
