import pandas as pd
import numpy as np
import re

# --- 1. DATOS DE EJEMPLO ---

# DataFrame de trabajo (el que quieres mapear)
df = pd.read_csv('var_nofa_second.tsv', sep='\t', index_col=False)



# Diccionario/Tabla de REFERENCIA MedDRA (¡DEBES REEMPLAZAR ESTO!)
# Simula la tabla de consulta donde 'Name' es el término (LLT o PT)
# que encuentras en tus datos y 'PT_Name' es el término preferido estandarizado.
meddra_df = pd.read_csv('meddra_all_se.tsv', sep='\t', index_col=False, header=None)
meddra_df.columns = ['CID_1', 'CID_2', 'Term_ID', 'Term_Level', 'PT_ID', 'Term_Name']


df_llt_to_pt_id = meddra_df[meddra_df['Term_Level'] == 'LLT']
dict_llt_name_to_pt_id = df_llt_to_pt_id.set_index('Term_Name')['PT_ID'].to_dict()

df_pt_info = meddra_df[meddra_df['Term_Level'] == 'PT']
dict_pt_id_to_pt_name = df_pt_info.set_index('PT_ID')['Term_Name'].to_dict()    

df_llt_to_pt_id['PT_Name'] = df_llt_to_pt_id['PT_ID'].map(dict_pt_id_to_pt_name)

dict_llt_name_to_pt_name = df_llt_to_pt_id.set_index('Term_Name')['PT_Name'].to_dict()

# ==============================================================================
# 2. FUNCIÓN AVANZADA DE MAPEO DE CELDAS MULTIVALOR
# ==============================================================================

def map_multivalue_cell(cell_value, mapping_dict, is_id_mapping=False):
    """
    Procesa una celda con múltiples términos separados por coma y aplica el mapeo.

    Args:
        cell_value (str): Valor de la celda (ej. "Side Effect:Vomiting, Disease:Hepatitis C").
        mapping_dict (dict): Diccionario de mapeo (LLT Name -> PT Name o PT ID).
        is_id_mapping (bool): True si el mapeo es para IDs (para manejar 'NaN' correctamente).

    Returns:
        str: Cadena de términos/IDs mapeados separados por coma, o NaN si la celda es inválida.
    """
    if pd.isna(cell_value) or cell_value.strip() == "":
        return np.nan

    # 1. Separar los términos por coma
    terms = cell_value.split(',')
    mapped_results = []
    
    # 2. Procesar cada término
    for term in terms:
        term = term.strip()
        
        # 3. Limpiar prefijos (ej. "Side Effect:Vomiting" -> "Vomiting")
        # Esto elimina cualquier texto antes del primer ':' (incluyendo el ':')
        
        cat_term = re.split(':', term)[0].strip()
        cleaned_term = re.sub(r'^[^:]+:\s*', '', term).strip()
        
        
        if cleaned_term:            
            mapped_value = cat_term + ':' + (mapping_dict.get(cleaned_term)) if mapping_dict.get(cleaned_term) and isinstance(mapping_dict.get(cleaned_term), str) else None
            

            if mapped_value:
                # Si el valor mapeado existe, añádelo
                mapped_results.append(str(mapped_value))
            elif not is_id_mapping:
                # Si no se encuentra mapeo para el NOMBRE (y no estamos buscando ID),
                # mantenemos el término original limpiado para evitar perder datos.
                # NOTA: Esto solo ocurre si un LLT no está en la tabla de MedDRA.
                mapped_results.append(cat_term + ':' + cleaned_term)

    # 5. Devolver los resultados combinados
    if mapped_results:
        # Usamos set() para asegurar que los PT e IDs sean únicos, luego lo unimos con ', '
        return ', '.join(sorted(list(set(mapped_results))))
    else:
        return np.nan # Devolver NaN si no se pudo mapear nada útil


# ==============================================================================
# 3. DATAFRAME DE TRABAJO (df) Y EJECUCIÓN DEL MAPEO
# ==============================================================================




# --- Aplicar el mapeo al Effect_phenotype ---
col_prefix = 'Effect_phenotype'

# A) Mapear a NOMBRE PT (reemplaza la columna original)
df[col_prefix] = df[col_prefix].apply(
    lambda x: map_multivalue_cell(x, dict_llt_name_to_pt_name, is_id_mapping=False)
)

# B) Mapear a ID PT (crea la nueva columna de ID)
df[f'{col_prefix}_id'] = df[col_prefix].apply(
    lambda x: map_multivalue_cell(x, dict_llt_name_to_pt_id, is_id_mapping=True)
)


# --- Aplicar el mapeo al Pop_Phenotypes/Diseases ---
col_prefix = 'Pop_Phenotypes/Diseases'

# A) Mapear a NOMBRE PT (reemplaza la columna original)
df[col_prefix] = df[col_prefix].apply(
    lambda x: map_multivalue_cell(x, dict_llt_name_to_pt_name, is_id_mapping=False)
)

# B) Mapear a ID PT (crea la nueva columna de ID)
df[f'{col_prefix}_id'] = df[col_prefix].apply(
    lambda x: map_multivalue_cell(x, dict_llt_name_to_pt_id, is_id_mapping=True)
)


df = df.reindex(columns=['ATC', 'Drug', 'Variant/Haplotypes', 'Gene', 'Alleles', 
                    'Phenotype_outcome', 'Effect_direction', 'Effect_type', 
                    'Effect_phenotype', 'Effect_phenotype_id', 
                    'Metabolizer types', 'Population types', 
                    'Pop_Phenotypes/Diseases', 'Pop_Phenotypes/Diseases_id',
                    'Comparison Allele(s) or Genotype(s)', 'Comparison Metabolizer types', 
                    'Notes', 'Sentence', 
                    'Variant Annotation ID',
])


df.to_csv('var_nofa_second_PT.tsv', sep='\t', index=False)

