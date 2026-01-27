"""
import polars as pl
from rapidfuzz import process, fuzz, utils
import re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import multiprocessing

# 1. Carga ultra-rápida y preparación del diccionario con Polars
print("Cargando y procesando diccionario...")
df_dict = pl.read_csv("Drug_Compount_processed.tsv", separator="\t", columns=["Compound_CID", "Name", "Synonyms"])

# Separar sinónimos y crear una tabla maestra (Term -> CID)
# Esto reemplaza el bucle for lento de Python
df_expanded = (
    df_dict
    .select([
        pl.col("Compound_CID"),
        pl.col("Name"),
        pl.col("Synonyms").str.split(r'[|,\/]') # Regex split nativo de Polars
    ])
    .explode("Synonyms") # Expande la lista a filas
)

# Unimos Nombres y Sinónimos en una sola columna de búsqueda "Term"
# Normalizamos a minúsculas para mejorar el exact match inicial
reference_df = (
    pl.concat([
        df_expanded.select([pl.col("Name").alias("Term"), pl.col("Compound_CID")]),
        df_expanded.select([pl.col("Synonyms").alias("Term"), pl.col("Compound_CID")])
    ])
    .filter(pl.col("Term").is_not_null() & (pl.col("Term") != ""))
    .unique(subset=["Term"]) # Eliminamos duplicados
    .with_columns(pl.col("Term").str.strip_chars())
)

# Convertimos a diccionarios/listas para RapidFuzz (solo lo necesario)
ref_dict = dict(zip(reference_df["Term"].to_list(), reference_df["Compound_CID"].to_list()))
choices_list = list(ref_dict.keys()) # Lista estática para pasar a RapidFuzz (Crucial para velocidad)

print(f"Total unique terms to match against: {len(choices_list)}")

# ---------------------------------------------------------

# 2. Carga de datos a mapear
data_df = pl.read_csv("train_data/final_enriched_data.tsv", separator="\t")

# 3. ESTRATEGIA HÍBRIDA: Exact Match primero (O(1)) -> Fuzzy después (O(N))

# Paso A: Intento de Mapeo Exacto (Milisegundos)
# Hacemos un join directo. Esto resolverá el 60-80% de casos sin usar fuzzy.
data_df = data_df.with_columns(pl.col("Drug").alias("Drug_Clean").str.strip_chars())
data_df = data_df.join(reference_df, left_on="Drug_Clean", right_on="Term", how="left")
data_df = data_df.rename({"Compound_CID": "Drug_CID"})

# Identificar qué filas fallaron el exact match
missing_mask = data_df["Drug_CID"].is_null()
missing_drugs = data_df.filter(missing_mask)["Drug"].to_list()
unique_missing_drugs = list(set(missing_drugs)) # Solo procesamos los únicos que faltan

print(f"Drugs resueltos por Exact Match: {len(data_df) - len(missing_drugs)}")
print(f"Drugs pendientes para Fuzzy Match: {len(unique_missing_drugs)}")

# ---------------------------------------------------------

# 4. Función de Mapeo Fuzzy Optimizada
def get_best_match_batch(queries):
    """'''
    Función optimizada para correr en paralelo.
    Recibe un lote de queries y devuelve tuplas (query, cid_encontrado).
    '''"""
    results = []
    for query in queries:
        if not query or not isinstance(query, str):
            results.append((query, "No_match"))
            continue
            
        # Limpieza básica del query (replicando tu lógica de split pero simplificada)
        # Token set ratio ya maneja bien el desorden, no hace falta splitear manualmente tanto
        clean_query = query.replace('|', ' ').replace(',', ' ').replace('/', ' ')
        
        # Una sola búsqueda con el umbral más bajo aceptable (90)
        # process.extractOne es mucho más rápido si 'choices' es una lista pre-procesada, no un dict_keys
        match = process.extractOne(
            query=clean_query,
            choices=choices_list,
            scorer=fuzz.token_set_ratio,
            score_cutoff=90,
            processor=utils.default_process # Pre-procesamiento vectorizado interno de C++
        )
        
        if match:
            # match es (string_encontrado, score, indice)
            matched_term = match[0]
            results.append((query, ref_dict[matched_term]))
        else:
            results.append((query, "No_match"))
    return results

# ---------------------------------------------------------

# 5. Ejecución Paralela (Multiprocessing)
# RapidFuzz libera el GIL, pero para listas muy grandes, ProcessPoolExecutor escala mejor.

if len(unique_missing_drugs) > 0:
    print("Iniciando Fuzzy Match paralelo...")
    
    # Dividir el trabajo en chunks según número de CPUs
    num_workers = multiprocessing.cpu_count()
    chunks = np.array_split(unique_missing_drugs, num_workers)
    
    fuzzy_results_map = {}
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Mapeamos la función sobre los chunks
        results_lists = list(executor.map(get_best_match_batch, chunks))
        
        # Aplanamos resultados y creamos un diccionario de búsqueda rápida
        for result_batch in results_lists:
            for query, cid in result_batch:
                fuzzy_results_map[query] = cid

    # 6. Aplicar resultados al DataFrame original
    # Usamos map_dict de polars para llenar solo los huecos
    
    # Creamos una expresión condicional: Si ya tiene CID (del exact match), déjalo.
    # Si no, búscalo en el mapa de resultados fuzzy.
    
    # Convertimos el mapa a un DataFrame pequeño para hacer join (más rápido que map_dict en grandes volúmenes)
    fuzzy_df = pl.DataFrame({
        "Drug": list(fuzzy_results_map.keys()),
        "Fuzzy_CID": list(fuzzy_results_map.values())
    })
    
    data_df = data_df.join(fuzzy_df, on="Drug", how="left")
    
    # Coalesce: Toma Drug_CID (exacto), si es nulo toma Fuzzy_CID, si es nulo pon "No_match"
    data_df = data_df.with_columns(
        pl.coalesce([pl.col("Drug_CID"), pl.col("Fuzzy_CID")]).fill_null("No_match").alias("Drug_CID_Final")
    )
else:
    data_df = data_df.with_columns(pl.col("Drug_CID").fill_null("No_match").alias("Drug_CID_Final"))

# Limpieza final y guardado
final_df = data_df.select([
    pl.exclude(["Drug_CID", "Fuzzy_CID", "Drug_Clean", "Term", "Compound_CID", "Compound_CID_right"])
]).rename({"Drug_CID_Final": "Drug_CID"})

print("Guardando resultados...")
final_df.write_csv("final_mapped_data.tsv", separator="\t")
print("Proceso finalizado.")
"""
import polars as pl
from rapidfuzz import process, fuzz, utils
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial

# 1. Definimos la función de lógica PURA fuera (sin depender de variables globales)
def get_best_match_batch(queries, choices_list, ref_dict):
    """
    Esta función ahora recibe las opciones y el diccionario como argumentos
    para que funcione correctamente en los sub-procesos de Windows.
    """
    results = []
    for query in queries:
        if not query or not isinstance(query, str):
            results.append((query, "No_match"))
            continue
            
        clean_query = query.replace('|', ' ').replace(',', ' ').replace('/', ' ')
        
        # Procesamiento
        match = process.extractOne(
            query=clean_query,
            choices=choices_list,
            scorer=fuzz.token_set_ratio,
            score_cutoff=90,
            processor=utils.default_process
        )
        
        if match:
            # match es (string_encontrado, score, indice)
            matched_term = match[0]
            # Usamos el ref_dict que nos han pasado
            results.append((query, ref_dict.get(matched_term, "No_match")))
        else:
            results.append((query, "No_match"))
    return results

def main():
    # --- TODO EL CÓDIGO DE EJECUCIÓN DEBE IR AQUÍ DENTRO ---
    
    # 2. Carga y preparación (Solo ocurre una vez en el proceso principal)
    print("Cargando y procesando diccionario...")
    df_dict = pl.read_csv("Drug_Compount_processed.tsv", separator="\t", columns=["Compound_CID", "Name", "Synonyms"])

    df_expanded = (
        df_dict
        .select([
            pl.col("Compound_CID"),
            pl.col("Name"),
            pl.col("Synonyms").str.split(r'[|,\/]') 
        ])
        .explode("Synonyms") 
    )

    reference_df = (
        pl.concat([
            df_expanded.select([pl.col("Name").alias("Term"), pl.col("Compound_CID")]),
            df_expanded.select([pl.col("Synonyms").alias("Term"), pl.col("Compound_CID")])
        ])
        .filter(pl.col("Term").is_not_null() & (pl.col("Term") != ""))
        .unique(subset=["Term"]) 
        .with_columns(pl.col("Term").str.strip_chars())
    )

    # Convertimos a objetos de Python para pasar a los workers
    ref_dict = dict(zip(reference_df["Term"].to_list(), reference_df["Compound_CID"].to_list()))
    choices_list = list(ref_dict.keys()) 

    print(f"Total unique terms to match against: {len(choices_list)}")

    # 3. Carga de datos a mapear
    data_df = pl.read_csv("train_data/final_enriched_data.tsv", separator="\t")
    
    # Normalización previa
    data_df = data_df.with_columns(pl.col("Drug").alias("Drug_Clean").str.strip_chars())
    
    # Paso A: Exact Match
    data_df = data_df.join(reference_df, left_on="Drug_Clean", right_on="Term", how="left")
    data_df = data_df.rename({"Compound_CID": "Drug_CID"})

    missing_mask = data_df["Drug_CID"].is_null()
    missing_drugs = data_df.filter(missing_mask)["Drug"].to_list()
    unique_missing_drugs = list(set(missing_drugs))

    print(f"Drugs resueltos por Exact Match: {len(data_df) - len(missing_drugs)}")
    print(f"Drugs pendientes para Fuzzy Match: {len(unique_missing_drugs)}")

    # 4. Ejecución Paralela Protegida
    if len(unique_missing_drugs) > 0:
        print("Iniciando Fuzzy Match paralelo...")
        
        # Preparamos la función "congelando" los argumentos constantes (choices y dict)
        # Esto permite pasar la data pesada a los workers de forma segura
        func_con_datos = partial(get_best_match_batch, choices_list=choices_list, ref_dict=ref_dict)
        
        num_workers = multiprocessing.cpu_count()
        # Dividimos la lista de drugs en chunks
        chunks = np.array_split(unique_missing_drugs, num_workers)
        
        fuzzy_results_map = {}
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Ejecutamos
            results_lists = list(executor.map(func_con_datos, chunks))
            
            for result_batch in results_lists:
                for query, cid in result_batch:
                    fuzzy_results_map[query] = cid

        # 5. Unificación de resultados
        fuzzy_df = pl.DataFrame({
            "Drug": list(fuzzy_results_map.keys()),
            "Fuzzy_CID": list(fuzzy_results_map.values())
        })
        
        data_df = data_df.join(fuzzy_df, on="Drug", how="left")
        
        data_df = data_df.with_columns(
            pl.coalesce([pl.col("Drug_CID"), pl.col("Fuzzy_CID")]).fill_null("No_match").alias("Drug_CID_Final")
        )
    else:
        data_df = data_df.with_columns(pl.col("Drug_CID").fill_null("No_match").alias("Drug_CID_Final"))

    # Guardado
    final_df = data_df.select([
        pl.exclude(["Drug_CID", "Fuzzy_CID", "Drug_Clean", "Term", "Compound_CID", "Compound_CID_right"])
    ]).rename({"Drug_CID_Final": "Drug_CID"})

    print("Guardando resultados...")
    final_df.write_csv("final_mapped_data.tsv", separator="\t")
    print("Proceso finalizado con éxito.")

# --- PUNTO DE ENTRADA CRÍTICO PARA WINDOWS ---
if __name__ == "__main__":
    multiprocessing.freeze_support() # Opcional pero recomendado en scripts complejos
    main()