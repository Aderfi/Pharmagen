"""
etl_pipeline.py

Advanced Data Engineering Pipeline for Pharmagen (v3.0).
Feature Enriched: Drugs (DCI) + Genes + Context (Disease).
Multi-Task Targets: Category + Direction.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = Path("data/raw/variantAnnotations")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "final_training_data.tsv"

# --- HELPER FUNCTIONS ---

def normalize_drug_names(text: str) -> str:
    """Normaliza fármacos a base activa (DCI)."""
    if pd.isna(text): return "unknown"
    clean = str(text).lower().strip()
    salts_to_remove = [
        r'\bhydrochloride\b', r'\bhydrobromide\b', r'\bhydrochlorid\b',
        r'\bsodium\b', r'\bpotassium\b', r'\bcalcium\b', r'\blithium\b',
        r'\bsulfate\b', r'\bsulphate\b', r'\bphosphate\b', r'\bacetate\b',
        r'\bmaleate\b', r'\btartrate\b', r'\bcitrate\b', r'\bfumarate\b',
        r'\bmesylate\b', r'\bsuccinate\b', r'\bnitrate\b', r'\boxide\b',
        r'\btrihydrate\b', r'\bdihydrate\b', r'\bmonohydrate\b',
        r'\bdisodium\b', r'\bdipotassium\b', r'\bhcl\b'
    ]
    salt_pattern = '|'.join(salts_to_remove)
    clean = re.sub(salt_pattern, '', clean)
    return clean.strip()

def normalize_disease_context(text: str | None) -> str:
    """Limpia el contexto de enfermedad."""
    if pd.isna(text) or str(text).lower() in ["nan", "none", ""]:
        return "general_population"
    
    t = str(text).lower().strip()
    # Limpieza básica de ruido común en PharmGKB
    t = re.sub(r'^patients with\s+', '', t)
    t = re.sub(r'^people with\s+', '', t)
    t = re.sub(r'^subjects with\s+', '', t)
    t = re.sub(r'^individuals with\s+', '', t)
    return t.strip()

def clean_text(text: str | None) -> str:
    if pd.isna(text): return "unknown"
    return str(text).lower().strip()

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Loading raw datasets...")
    # Columnas clave incluyendo el contexto
    cols = ["Variant/Haplotypes", "Gene", "Drug(s)", "Phenotype Category", 
            "Significance", "Direction of effect", "Population Phenotypes or diseases"]
    
    try:
        # Nota: Population Phenotypes suele estar en var_drug_ann y var_pheno_ann
        df_drug = pd.read_csv(RAW_DIR / "var_drug_ann.tsv", sep="\t", usecols=lambda c: c in cols or c == "Variant Annotation ID", low_memory=False)
        df_pheno = pd.read_csv(RAW_DIR / "var_pheno_ann.tsv", sep="\t", usecols=lambda c: c in cols or c == "Variant Annotation ID", low_memory=False)
        df_fa = pd.read_csv(RAW_DIR / "var_fa_ann.tsv", sep="\t", low_memory=False) 
        
        return df_drug, df_pheno, df_fa
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

def explode_drugs(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Exploding & Normalizing drugs...")
    # Split
    df["Drug(s)"] = df["Drug(s)"].astype(str).apply(lambda x: re.split(r'[,;]|\s\+\s', x))
    # Explode
    df_exploded = df.explode("Drug(s)")
    # Normalize
    df_exploded["Drug(s)"] = df_exploded["Drug(s)"].apply(normalize_drug_names)
    # Filter empty
    df_exploded = df_exploded[~df_exploded["Drug(s)"].isin(["nan", "none", "", "unknown"])]
    return df_exploded

# --- TARGET ENGINEERING ---

def get_phenotype_category(row: pd.Series) -> str:
    """Target 1: ¿Qué tipo de efecto es?"""
    cat = clean_text(row.get("Phenotype Category", ""))
    sig = clean_text(row.get("Significance", ""))
    
    # Filtro de significancia
    if sig in ["no", "none", "not associated"]:
        return "No_Association"

    if "toxicity" in cat or "side effect" in cat: return "Toxicity"
    if "efficacy" in cat: return "Efficacy"
    if "metabolism" in cat or "pk" in cat: return "Metabolism_PK"
    if "dosage" in cat: return "Dosage"
    
    # Si es significativo pero no cae en categoría clara
    if sig == "yes": return "Association_General"
    
    return "No_Association"

def get_direction(row: pd.Series) -> str:
    """Target 2: ¿Aumenta o Disminuye?"""
    d = clean_text(row.get("Direction of effect", ""))
    sig = clean_text(row.get("Significance", ""))
    
    if sig in ["no", "none", "not associated"]: return "None"
    
    if any(x in d for x in ["increased", "higher", "faster", "greater"]): return "Increased"
    if any(x in d for x in ["decreased", "lower", "slower", "reduced"]): return "Decreased"
    
    return "Associated" # Dirección no especificada pero existe relación

def get_phenotype_detail(row: pd.Series) -> str:
    """
    Target 3: Contexto Específico (Fusión de 'Side effect/efficacy/other' + 'Phenotype').
    Ej: "Side Effect: Myopathy"
    """
    # Intentamos obtener la categoría específica de pheno_ann
    # Nota: Esta columna solo existe en var_pheno_ann.tsv, en otros archivos será NaN
    category_spec = clean_text(row.get("Side effect/efficacy/other", ""))
    phenotype_desc = clean_text(row.get("Phenotype", ""))
    
    # Si viene de drug_ann o fa_ann, estas columnas pueden estar vacías o ser distintas
    # Fallback a Phenotype Category si no hay info específica
    if category_spec == "unknown" and phenotype_desc == "unknown":
        return clean_text(row.get("Phenotype Category", "unknown"))

    # Construcción del detalle
    # Limpieza básica de ruido en Phenotype ("risk of", "severity of"...)
    phenotype_desc = re.sub(r'^(risk|severity|likelihood) of\s+', '', phenotype_desc)
    
    if category_spec != "unknown":
        # Formato: "Side Effect: Myopathy"
        # Si la descripción ya contiene la categoría, no la duplicamos
        if category_spec in phenotype_desc:
            return phenotype_desc
        return f"{category_spec}: {phenotype_desc}"
    
    return phenotype_desc

def run_etl():
    # 1. Load
    df_drug, df_pheno, df_fa = load_data()
    
    # 2. Unify
    # Asegurar columnas necesarias para el nuevo target
    for df in [df_drug, df_pheno]:
        if "Side effect/efficacy/other" not in df.columns: df["Side effect/efficacy/other"] = "nan"
        if "Phenotype" not in df.columns: df["Phenotype"] = "nan"

    # Asegurar que la columna de contexto existe en ambos (rellenar si falta)
    if "Population Phenotypes or diseases" not in df_drug.columns: df_drug["Population Phenotypes or diseases"] = "nan"
    if "Population Phenotypes or diseases" not in df_pheno.columns: df_pheno["Population Phenotypes or diseases"] = "nan"
    
    df_main = pd.concat([df_drug, df_pheno], ignore_index=True)
    
    # 3. Explode Drugs
    df_main = explode_drugs(df_main)
    
    # 4. Clean Context (Disease)
    logger.info("Normalizing Disease Context...")
    df_main["Disease_Context"] = df_main["Population Phenotypes or diseases"].apply(normalize_disease_context)
    
    # 5. Enrich (Functional)
    # ... (Misma lógica FA que antes) ...
    
    # 6. Target Engineering (Targets Separados)
    logger.info("Engineering Targets (Category + Direction + Detail)...")
    df_main["Target_Type"] = df_main.apply(get_phenotype_category, axis=1)
    df_main["Target_Direction"] = df_main.apply(get_direction, axis=1)
    df_main["Target_Detail"] = df_main.apply(get_phenotype_detail, axis=1)
    
    # 7. Renaming
    rename_map = {
        "Variant/Haplotypes": "Variant_ID",
        "Gene": "Gene_ID",
        "Drug(s)": "Drug_ID"
    }
    df_main = df_main.rename(columns=rename_map)
    df_main = df_main.fillna("unknown")
    
    # 8. Aggregation & One-Hot
    logger.info("Aggregating & Binarizing...")
    
    # Clave única
    group_keys = ["Variant_ID", "Gene_ID", "Drug_ID", "Disease_Context"]
    
    df_grouped = df_main.groupby(group_keys).agg({
        "Target_Type": lambda x: list(set(x)),
        "Target_Direction": lambda x: list(set(x)),
        "Target_Detail": lambda x: list(set(x)) # Nuevo
    }).reset_index()
    
    # Binarizar Type
    mlb_type = MultiLabelBinarizer()
    type_bin = mlb_type.fit_transform(df_grouped["Target_Type"])
    type_cols = [f"Type_{c}" for c in mlb_type.classes_]
    df_type = pd.DataFrame(type_bin, columns=type_cols)
    
    # Binarizar Direction
    mlb_dir = MultiLabelBinarizer()
    dir_bin = mlb_dir.fit_transform(df_grouped["Target_Direction"])
    dir_cols = [f"Dir_{c}" for c in mlb_dir.classes_]
    df_dir = pd.DataFrame(dir_bin, columns=dir_cols)
    
    # NO Binarizamos Detail masivamente (serían miles de columnas)
    # Lo dejamos como lista de strings para análisis o embeddings de texto futuros
    # Opcional: Podríamos binarizar solo los Top 50 más comunes si quieres.
    # Por ahora, lo guardamos "raw" para no explotar la memoria.
    
    # Final Join
    df_final = pd.concat([df_grouped[group_keys + ["Target_Detail"]], df_type, df_dir], axis=1)
    
    # Export
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, sep="\t", index=False)
    
    logger.info(f"✅ ETL v3.1 Complete. Shape: {df_final.shape}")
    logger.info(f"Input Features: {group_keys}")
    logger.info(f"Targets: {len(type_cols)} Types + {len(dir_cols)} Directions + Detail (Raw)")

if __name__ == "__main__":
    run_etl()
