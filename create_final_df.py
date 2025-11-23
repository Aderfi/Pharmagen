import polars as pl
import pandas as pd
import re
from typing import List, Optional

from rapidfuzz import process, fuzz 

from src.utils.data import load_atc_dictionary, drugs_to_atc, normalize_effect_type, normalize_variant_input, clean_drugs_col, new_coso

METABOLIZER_MAPPING = {
    # --- Estándar (CYP450) ---
    'poor metabolizer': 'PM',
    'intermediate metabolizer': 'IM',
    'normal metabolizer': 'NM',
    'extensive metabolizer': 'NM', # Sinónimo histórico
    'rapid metabolizer': 'RM',
    'ultrarapid metabolizer': 'UM',
    
    # --- Acetiladores (NAT2) ---
    'slow acetylator': 'PM',         # Lento = Acumula fármaco (Tox) -> PM
    'intermediate acetylator': 'IM',
    'rapid acetylator': 'RM',        # Rápido = Elimina fármaco -> RM
    
    # --- Actividad Funcional ---
    'deficiency': 'PM',              # Ausencia de función
    'low activity': 'PM',            # Baja función
    'intermediate activity': 'IM',
    'high activity': 'UM',           # Alta función -> UM
    'reduced metabolizers': 'PM',    # Conservador: tratar como Pobre
    
    # --- Nulos ---
    'none': 'Unknown',
    'nan': 'Unknown',
    'unknown': 'Unknown'
}

# --- Configuración y Constantes ---
file = "train_data/kukafinal_test_filled.tsv"
json_file = "data/dicts/ATC_drug_med.json"
UNK_TOKEN = "__UNKNOWN__"

col_mapping = {
    "Drug": "Drug",
    "Gene": "Gene_Symbol",
    "Allele": "Allele",
    "Variant/Haplotypes": "Variant/Haplotypes",
    "Metabolizer types": "Metabolizer_status",
    "Phenotype_outcome": "Phenotype_outcome",
    "Effect_direction": "Effect_direction",
    "Effect_type": "Effect_type",
    "Pop_phenotype_id": "Previous_Condition_Term"
}

list_cols = list(col_mapping.keys())


medcat = "BACKUPS/meddra_all_se.tsv"
medcat_df = pd.read_csv(medcat, sep="\t", encoding="utf-8", usecols=[5, 2], names=["Dis_name", "MedDRA_Code"], header=0)

meddra_dict = {k: v for k, v in zip(medcat_df["Dis_name"].str.lower(), medcat_df["MedDRA_Code"])}
del medcat_df
############################ DESDE AQUI TOT BIEN ############################
#df = pl.read_csv(file, separator="\t", encoding="utf-8").to_pandas()
df = pd.read_csv(file, sep="\t", encoding="utf-8", usecols=list_cols)
############################ HASTA AQUI TOT BIEN ############################
def map_disease_to_meddra(
    df: pd.DataFrame, 
    meddra_mapping, 
    col_name: str = "Pop_Phenotypes/Diseases"
) -> pd.DataFrame:
    """
    Mapea términos de enfermedades a códigos MedDRA manejando multi-etiquetas y prefijos.
    
    Args:
        df: DataFrame de entrada.
        meddra_mapping: Diccionario {Nombre: Codigo}.
        col_name: Nombre de la columna a procesar.
        
    Returns:
        DataFrame con la columna mapeada.
    """
    df_out = df.copy()
    
    # 1. Optimización: Normalizar claves del diccionario para búsqueda O(1)
    # Convertimos todo a minúsculas para evitar problemas de case-sensitivity
    meddra_lower = {k.lower(): v for k, v in meddra_mapping.items()}
    meddra_keys = list(meddra_lower.keys()) # Lista para fuzzy matching
    
    # 2. Caché local para evitar re-calcular fuzzy matches repetidos
    memoization = {}

    def _resolve_single_term(term_str: str) -> str:
        """Procesa un solo término (ej: 'Disease:Hypercholesterolemia')."""
        term_str = term_str.strip()
        
        # Caso base: Valores vacíos o marcadores de "sin datos"
        if not term_str or term_str in ["No_Prev_Pheno/Diseases", "nan", "None"]:
            return term_str

        # Separar Categoría y Nombre (ej: "Disease" : "Epilepsy")
        if ":" in term_str:
            parts = term_str.split(":", 1)
            category, name_raw = parts[0], parts[1]
        else:
            # Si no tiene formato 'Cat:Nombre', asumimos que todo es nombre
            category, name_raw = "Other", term_str

        name_clean = name_raw.strip().lower()
        
        # --- Lógica de Mapeo ---
        
        # A. Verificar Caché
        if name_clean in memoization:
            code = memoization[name_clean]
        
        # B. Búsqueda Exacta
        elif name_clean in meddra_lower:
            code = meddra_lower[name_clean]
            memoization[name_clean] = code # Guardar en caché
            
        # C. Búsqueda Fuzzy (Solo si falla la exacta)
        else:
            # score_cutoff=85 es un buen equilibrio, ajústalo si es muy estricto
            match = process.extractOne(
                name_clean, meddra_keys, scorer=fuzz.WRatio, score_cutoff=85
            )
            
            if match:
                matched_name = match[0]
                code = meddra_lower[matched_name]
            else:
                code = None # No se encontró
            
            memoization[name_clean] = code # Guardar resultado (incluso si es None)

        # Construir string de retorno
        if code:
            return f"{category}:{code}"
        else:
            # Si no hay mapeo, devolvemos el original o marcamos como desconocido
            return f"{category}:{name_raw}"

    def _process_row(row_val) -> str:
        """Maneja la celda completa, incluyendo separación por comas."""
        if pd.isna(row_val):
            return str(row_val)
            
        row_str = str(row_val)
        
        # 1. Separar por comas (Manejo Multi-Label)
        # "Disease:A, Other:B" -> ["Disease:A", " Other:B"]
        terms = [t for t in row_str.split(",") if t.strip()]
        
        # 2. Resolver cada término individualmente
        mapped_terms = [_resolve_single_term(t) for t in terms]
        
        # 3. Volver a unir (Usando pipe '|' es más seguro para CSVs que la coma)
        # Si prefieres coma, cambia "|" por ","
        return "|".join(mapped_terms)

    # Aplicar la transformaciónvectorizada (mucho más rápido que iterar con for)
    df_out[col_name] = df_out[col_name].apply(_process_row)
    
    return df_out


def normalize_metabolizer_status(val: Optional[str]) -> List[str]:
    """
    Convierte strings complejos como 'poor metabolizer and ultrarapid metabolizer'
    en una lista de códigos estandarizados ['PM', 'UM'].
    """
    if pd.isna(val) or str(val).lower() in ['none', 'nan', '']:
        return ['Unknown']

    val_str = str(val).lower().strip()
    
    # 1. Separar por ' and ', ' + ', ',', '|'
    # Esto rompe "intermediate metabolizer and poor metabolizer" en dos partes
    parts = re.split(r'\s+(?:and|\+|\||,)\s+', val_str)
    
    codes = set()
    for part in parts:
        part = part.strip()
        if not part: continue
        
        # Buscar en el diccionario
        if part in METABOLIZER_MAPPING:
            codes.add(METABOLIZER_MAPPING[part])
        else:
            # Fallback: Si algo no está en el mapa, intenta buscar palabras clave
            if 'poor' in part or 'slow' in part: codes.add('PM')
            elif 'interm' in part: codes.add('IM')
            elif 'ultra' in part: codes.add('UM')
            elif 'rapid' in part: codes.add('RM')
            elif 'normal' in part or 'extens' in part: codes.add('NM')
            else:
                codes.add('Unknown')
                
    return list(codes)


# Mapeo para normalización
mapper_metabolizer = {
    "poor metabolizer": "PM",
    "intermediate metabolizer": "IM",
    "extensive metabolizer": "EM",
    "ultrarapid metabolizer": "UM"     
}

# Mapeo de columnas (Renombrado)
col_mapping = {
    "Drug": "Drug",
    "Gene": "Gene_Symbol",
    "Allele": "Allele",
    "Variant/Haplotypes": "Variant/Haplotypes",
    "Metabolizer types": "Metabolizer_status",
    "Phenotype_outcome": "Phenotype_outcome",
    "Effect_direction": "Effect_direction",
    "Effect_type": "Effect_type",
    "Pop_phenotype_id": "Previous_Condition_Term"
}

# --- 1. Carga y Pre-procesamiento Inicial ---
# Cargamos con Polars por velocidad, pero pasamos a Pandas para manipulación de strings compleja
#df_resultado = map_disease_to_meddra(df, meddra_dict)
#df = df_resultado.copy()

# Limpieza inicial
df['Gene'] = df['Gene'].replace("Intron/No_Especificado", "__INTRONIC__")

mapper_names ={
        "Farmaco_Desconocido/NoRelacionado": UNK_TOKEN,
        "4-hydroxytamoxifen": "4-hydroxytamoxifen(tamoxifen)",
        "mycophenolate mofetil": "mycophenolate mofetil(mycophenolic acid)",
        "clopidogrel carboxylic acid": "clopidogrel",
        "cotinine glucuronide": "cotinine",
        "4-hydroxytamoxifen, endoxifen": "tamoxifen, endoxifen",
        "acetylisoniazid": "acetylisoniazid(isoniazid)",
        "n-desmethyltramadol": "n-desmethyltramadol(tramadol)",
        "hydroxyamitriptyline": "hydroxyamitriptyline(amitriptyline)",
        "rifampin": "rifampicin",
        "n-desmethylclozapine": "n-desmethylclozapine(clozapine)",
        "5-hydroxy-omeprazole": "5-hydroxy-omeprazole(omeprazole)",
        "desmethylvenlafaxine": "desmethylvenlafaxine(venlafaxine)",
        "sertraline hydrochloride": "sertraline",
        "paroxetine hydrochloride": "paroxetine",
        "fluoxetine hydrochloride": "fluoxetine",
        "norfluoxetine": "norfluoxetine(fluoxetine)",
        "desvenlafaxine": "desvenlafaxine(venlafaxine)",
        "hydroxybupropion": "hydroxybupropion(bupropion)",
        "hydroxychloroquine sulfate": "hydroxychloroquine",
        "sn-38": "sn-38(irinotecan)",
        "amitriptyline hydrochloride": "amitriptyline",
        "venlafaxine hydrochloride": "venlafaxine",
        "clozapine hydrochloride": "clozapine",
        "bupropion hydrochloride": "bupropion",
        "isoniazid hydrazine": "isoniazid",
    }

atc_dict = load_atc_dictionary(json_file, 1)
if not all([k.islower() for k in list(atc_dict.keys())]):
    atc_dict = {k.lower(): v for k, v in atc_dict.items()}

df["Drug"] = df["Drug"].map(lambda x: mapper_names.get(x.lower()) if x in mapper_names and isinstance(x, str) else x)
df.rename(columns=col_mapping, inplace=True)

df["Drug"] = df["Drug"].apply(clean_drugs_col)

atc_keys_list = list(atc_dict.keys())
df["ATC"] = df["Drug"].apply(
    lambda x: new_coso(
        drug_cell=x, 
        atc_dict_rev=atc_dict, 
        atc_keys=atc_keys_list, 
        score_cutoff=90
    )
)
lista_metab = list(METABOLIZER_MAPPING.keys())    
for idx in df.index:
    for z in lista_metab:
        gene = str(df.at[idx, "Gene_Symbol"])
        valaplo = str(df.at[idx, "Variant/Haplotypes"])

        if re.search(gene, valaplo, re.IGNORECASE) and re.search(z, valaplo):
            df.at[idx, "Variant/Haplotypes"] = valaplo.strip(gene).strip()


# Explode de alelos (simplificado)
df['Allele'] = df['Allele'].str.split('+')
df = df.explode('Allele').reset_index(drop=True)
df['Allele'] = df['Allele'].str.strip()
# Seleccionamos y renombramos columnas

# --- 2. Enriquecimiento ATC ---


atc_df = df.copy()

# --- 3. Normalización y Limpieza Final ---


atc_df["Previous_Condition_Term"] = atc_df["Previous_Condition_Term"].replace("No_Prev_Pheno/Diseases", "Healthy")
atc_df["Allele"] = atc_df["Allele"].replace("Alelle_NoEspecificado", UNK_TOKEN)
atc_df["Previous_Condition_Term"] = atc_df["Previous_Condition_Term"].replace("No_Prev_Pheno/Diseases(ID:NA)", "Healthy")
#atc_df["Drug"] = atc_df["Drug"].replace("aspirin", "acetylsalicylic acid")
atc_df['Gene_Symbol'] = atc_df['Gene_Symbol'].replace("Intron/No_Especificado", "__INTRONIC__")

atc_df["Previous_Condition_Term"] = atc_df["Previous_Condition_Term"].replace(r", +", "|", regex=True)
atc_df["Previous_Condition_Term"] = atc_df["Previous_Condition_Term"].replace(r"\s+", "_", regex=True)



# Vectorización: Reemplaza el bucle 'for' por 'apply' para mayor eficiencia y limpieza
atc_df["Variant_Normalized"] = atc_df.apply(
    lambda row: normalize_variant_input(row["Gene_Symbol"], row["Allele"], row["Variant/Haplotypes"]), 
    axis=1
)



# Limpieza general de strings en todas las columnas de tipo objeto
obj_cols = atc_df.select_dtypes(include=['object']).columns
atc_df[obj_cols] = atc_df[obj_cols].apply(lambda x: x.str.strip())

# Limpieza Regex final y reemplazos
#atc_df['Variant_Normalized'] = atc_df['Variant_Normalized'].replace(r"[\[\']+ | [\]\']", "", regex=True)


atc_df_x = normalize_effect_type(atc_df, "Effect_type")
# --- 4. Salida ---

#print(atc_df_x.sample(10))

atc_df_x.to_csv("train_data/final_enriched_data.tsv", sep="\t", encoding="utf-8", index=False)


#atc_df_pl = pl.from_pandas(atc_df_x)

#print(atc_df_pl.sample(10))