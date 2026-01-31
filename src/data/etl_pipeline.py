"""
etl_pipeline.py

Advanced Data Engineering Pipeline for Pharmagen.
Converts raw PharmGKB annotations into a clean, ML-ready dataset.

Steps:
1. Ingest raw TSVs (Drug, Phenotype, Functional).
2. Unify Drug & Phenotype evidence tables.
3. Explode compound drugs (1-to-Many).
4. Engineer categorical Targets (Labels) with granular clinical logic.
5. Enrich with Functional Analysis context.
6. Aggregate results for Multi-Label Classification.
7. Export final training set.
"""

import pandas as pd
import logging
from pathlib import Path
import re

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
RAW_DIR = Path("data/raw/variantAnnotations")
PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "final_training_data.tsv"

def clean_text(text: str | None) -> str:
    """Standardizes text: lower, strip, remove special chars."""
    if pd.isna(text):
        return "unknown"
    t = str(text).lower().strip()
    return t

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads raw datasets with robust error handling."""
    logger.info("Loading raw datasets...")
    
    # Define columns to keep to ensure schema alignment between Drug and Phenotype files
    cols_drug = ["Variant/Haplotypes", "Gene", "Drug(s)", "Phenotype Category", "Significance", "Direction of effect", "Sentence"]
    cols_pheno = ["Variant/Haplotypes", "Gene", "Drug(s)", "Phenotype Category", "Significance", "Direction of effect", "Sentence"]
    
    try:
        df_drug = pd.read_csv(RAW_DIR / "var_drug_ann.tsv", sep="\t", usecols=lambda c: c in cols_drug or c == "Variant Annotation ID")
        df_pheno = pd.read_csv(RAW_DIR / "var_pheno_ann.tsv", sep="\t", usecols=lambda c: c in cols_pheno or c == "Variant Annotation ID")
        df_fa = pd.read_csv(RAW_DIR / "var_fa_ann.tsv", sep="\t") 
        
        logger.info(f"Loaded: Drug ({len(df_drug)}), Pheno ({len(df_pheno)}), FA ({len(df_fa)})")
        return df_drug, df_pheno, df_fa
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise

def explode_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles comma-separated drugs.
    Ex: "DrugA, DrugB" -> Row1(DrugA), Row2(DrugB)
    """
    logger.info("Exploding composite drug entries...")
    
    # Split by comma or semicolon
    df["Drug(s)"] = df["Drug(s)"].astype(str).apply(lambda x: re.split(r'[,;]\s*', x))
    
    # Explode
    df_exploded = df.explode("Drug(s)")
    
    # Clean drug names
    df_exploded["Drug(s)"] = df_exploded["Drug(s)"].apply(clean_text)
    
    # Remove rows with empty drugs or "nan"
    df_exploded = df_exploded[~df_exploded["Drug(s)"].isin(["nan", "none", ""])]
    
    logger.info(f"Rows after explosion: {len(df_exploded)}")
    return df_exploded

def engineer_target(row: pd.Series) -> str:
    """
    Logic Rule Engine to define Granular Clinical Outcomes.
    Converts raw metadata into specific predictive labels (Multi-Label ready).
    """
    cat = clean_text(row.get("Phenotype Category", ""))
    direction = clean_text(row.get("Direction of effect", ""))
    sig = clean_text(row.get("Significance", ""))
    
    # 1. Filter Non-Significant findings
    if sig in ["no", "none"] or "not associated" in str(row.get("Is/Is Not associated", "")).lower():
        return "No Association"

    # 2. Metabolism / PK (The "Mechanism")
    # Decreased PK often implies Accumulation -> Toxicity
    # Increased PK often implies Rapid Clearance -> Low Efficacy
    if "metabolism" in cat or "pk" in cat:
        if any(x in direction for x in ["decreased", "lower", "reduced"]):
            return "PK: Poor Metabolizer"
        if any(x in direction for x in ["increased", "higher", "faster"]):
            return "PK: Rapid Metabolizer"

    # 3. Toxicity / Side Effects (The "Adverse Outcome")
    if "toxicity" in cat or "side effect" in cat:
        if any(x in direction for x in ["increased", "higher", "associated"]):
            return "Toxicity: High Risk"
        if any(x in direction for x in ["decreased", "lower"]):
            return "Toxicity: Reduced Risk" # Protective factor

    # 4. Efficacy (The "Therapeutic Outcome")
    if "efficacy" in cat:
        if any(x in direction for x in ["decreased", "lower", "poor", "reduced"]):
            return "Efficacy: Poor Response" # Resistance
        if any(x in direction for x in ["increased", "higher", "better", "associated"]):
            return "Efficacy: High Response"

    # 5. Dosage (The "Actionable Insight")
    if "dosage" in cat:
        if any(x in direction for x in ["increased", "higher"]):
            return "Dosage: Higher Requirement" # Needs more drug
        if any(x in direction for x in ["decreased", "lower"]):
            return "Dosage: Lower Requirement" # Needs less drug (risk of overdose)

    # Fallback for vague but positive associations
    if sig == "yes" or "associated" in str(row.get("Is/Is Not associated", "")).lower():
        # Try to infer from category alone if direction is missing
        if "toxicity" in cat:
            return "Toxicity: Risk (Undefined)"
        if "efficacy" in cat:
            return "Efficacy: Altered"
        return "Association: General"

    return "No Association"

def run_etl():
    # 1. Load
    df_drug, df_pheno, df_fa = load_data()
    
    # 2. Unify (Concatenate)
    df_drug["_source"] = "drug_ann"
    df_pheno["_source"] = "pheno_ann"
    
    df_main = pd.concat([df_drug, df_pheno], ignore_index=True)
    logger.info(f"Unified dataset size: {len(df_main)}")
    
    # 3. Explode Drugs
    df_main = explode_drugs(df_main)
    
    # 4. Enrich with Functional Analysis (Left Join)
    df_fa_dedup = df_fa.drop_duplicates(subset=["Variant/Haplotypes", "Gene"])
    cols_to_add = ["Functional terms", "Assay type"]
    available_cols = [c for c in cols_to_add if c in df_fa_dedup.columns]
    
    df_merged = pd.merge(
        df_main,
        df_fa_dedup[["Variant/Haplotypes", "Gene"] + available_cols],
        on=["Variant/Haplotypes", "Gene"],
        how="left"
    )
    
    # 5. Target Engineering
    logger.info("Engineering Target Labels...")
    df_merged["Outcome_Label"] = df_merged.apply(engineer_target, axis=1)
    
    # 6. Final Cleaning & Renaming
    rename_map = {
        "Variant/Haplotypes": "Variant_ID",
        "Gene": "Gene_ID",
        "Drug(s)": "Drug_ID",
        "Functional terms": "Functional_Impact"
    }
    df_merged = df_merged.rename(columns=rename_map)
    df_merged = df_merged.fillna("unknown")
    
    # --- MULTI-LABEL AGGREGATION ---
    logger.info("Aggregating Outcomes for Multi-Label Strategy...")
    
    # Group by unique biological keys
    group_keys = ["Variant_ID", "Gene_ID", "Drug_ID"]
    
    # Collect all outcomes into a list for each unique combination
    df_grouped = df_merged.groupby(group_keys).agg({
        "Outcome_Label": lambda x: list(set(x)), # Unique list of labels
        "Functional_Impact": "first",            # Representative functional impact
        "_source": "count"                       # Track evidence count
    }).reset_index()
    
    df_grouped = df_grouped.rename(columns={"_source": "Evidence_Count"})
    
    # 7. Stratification Column (For CV)
    # Join sorted labels as a string for stratification
    df_grouped["_stratify"] = df_grouped["Outcome_Label"].apply(lambda x: str("_".join(sorted(x))))
    
    # Remove extremely rare strata (<5 samples)
    vc = df_grouped["_stratify"].value_counts()
    rare_strata = vc[vc < 5].index
    df_grouped.loc[df_grouped["_stratify"].isin(rare_strata), "_stratify"] = "other"
    
    # 8. Export
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    # Save as TSV. Note: Lists will be saved as string representations "['A', 'B']".
    df_grouped.to_csv(OUTPUT_FILE, sep="\t", index=False)
    
    logger.info("==========================================")
    logger.info(f"✅ ETL Complete. Output: {OUTPUT_FILE}")
    logger.info(f"Final shape: {df_grouped.shape}")
    logger.info(f"Unique Combinations: {len(df_grouped)}")
    logger.info("==========================================")

if __name__ == "__main__":
    run_etl()
