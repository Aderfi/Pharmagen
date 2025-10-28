# drugs.csv: "PharmGKB Accession Id","Name","Generic Names","Trade Names","Brand Mixtures","Type","Cross-references","SMILES","InChI","Dosing Guideline","External Vocabulary","Clinical Annotation Count","Variant Annotation Count","Pathway Count","VIP Count","Dosing Guideline Sources","Top Clinical Annotation Level","Top FDA Label Testing Level","Top Any Drug Label Testing Level","Label Has Dosing Info","RxNorm Identifiers","ATC Identifiers","PubChem Compound Identifiers","Top CPIC Pairs Level","FDA Label has Prescribing Info","In FDA PGx Association Sections"
# relationships.csv: "Entity1_id","Entity1_name","Entity1_type","Entity2_id","Entity2_name","Entity2_type","Evidence","Association","PK","PD","PMIDs"
# study_parameters.csv:"Study Parameters ID","Variant Annotation ID","Study Type","Study Cases","Study Controls","Characteristics","Characteristics Type","Frequency In Cases","Allele Of Frequency In Cases","Frequency In Controls","Allele Of Frequency In Controls","P Value","Ratio Stat Type","Ratio Stat","Confidence Interval Start","Confidence Interval Stop","Biogeographical Groups"


# var_drug_ann.csv: "Variant Annotation ID","Variant/Haplotypes","Gene","Drug(s)","PMID","Phenotype Category","Significance","Notes","Sentence","Alleles","Specialty Population","Metabolizer types","isPlural","Is/Is Not associated","Direction of effect","PD/PK terms","Multiple drugs And/or","Population types","Population Phenotypes or diseases","Multiple phenotypes or diseases And/or","Comparison Allele(s) or Genotype(s)","Comparison Metabolizer types"
# var_fa_ann.csv: "Variant Annotation ID","Variant/Haplotypes","Gene","Drug(s)","PMID","Phenotype Category","Significance","Notes","Sentence","Alleles","Specialty Population","Assay type","Metabolizer types","isPlural","Is/Is Not associated","Direction of effect","Functional terms","Gene/gene product","When treated with/exposed to/when assayed with","Multiple drugs And/or","Cell type","Comparison Allele(s) or Genotype(s)","Comparison Metabolizer types"
# var_pheno_ann.csv: "Variant Annotation ID","Variant/Haplotypes","Gene","Drug(s)","PMID","Phenotype Category","Significance","Notes","Sentence","Alleles","Specialty Population","Metabolizer types","isPlural","Is/Is Not associated","Direction of effect","Side effect/efficacy/other","Phenotype","Multiple phenotypes And/or","When treated with/exposed to/when assayed with","Multiple drugs And/or","Population types","Population Phenotypes or diseases","Multiple phenotypes or diseases And/or","Comparison Allele(s) or Genotype(s)","Comparison Metabolizer types"

import pandas as pd
import numpy as np
import re
import glob

# cols = Variant Annotation ID;Variant/Haplotypes;Gene;Drug(s);PMID;Phenotype Category;Significance;Notes;Sentence;Alleles;Specialty Population;Metabolizer types;isPlural;Association;Direction of effect;PD/PK terms;Multiple drugs And/or;Population types;Population Phenotypes or diseases;Multiple phenotypes or diseases And/or;Comparison Allele(s) or Genotype(s);Comparison Metabolizer types
# COL = ["Variant Annotation ID", "Variant/Haplotypes", "Gene", "Drug(s)", "Phenotype Category", "Alleles", "Direction of effect", "PD/PK terms", "Multiple drugs And/or", "Population types", "Population Phenotypes or diseases", "Multiple phenotypes or diseases And/or", "Sentence", "Notes"]

csv_files = glob.glob("var_*.csv")

df = pd.DataFrame()
COLUMNAS = [
    "Variant_id",
    "Drugs",
    "Gene",
    "Alleles",
    "Genotype",
    "Outcome_category",
    "Effect_direction",
    "Effect_category",
    "Effect_subcat",
    "Population_types",
    "Enzyme/Protein",
    "Metabolizer_types",
    "Population_diseases",
    "Sentence",
    "Notes",
]

col_map = {
    "Variant_id": "Variant_id",
    "Drugs": "Drugs",
    "Gene": "Gene",
    "Alleles": "Alleles",
    "Genotype": "Genotype",
    "Outcome_category": "Outcome_category",
    "Effect_direction": "Effect_direction",
    "Effect_category": "Effect_category",
    "Effect_subcat": "Effect_subcat",
    "Population_types": "Population_types",
    "Population_diseases": "Population_diseases",
    "Enzyme/Protein": "Enzyme/Protein",
    "Metabolizer_types": "Metabolizer_types",
    "Sentence": "Sentence",
    "Notes": "Notes",
}

df_list = []

for file in csv_files:
    temp_df = pd.read_csv(file, sep=";", dtype=str)
    # Renombra las columnas según el diccionario de mapeo si existen en el archivo
    temp_df = temp_df.rename(
        columns={c: col_map[c] for c in temp_df.columns if c in col_map}
    )
    # Agrega las columnas que falten y ordena según COLUMNAS
    for col in COLUMNAS:
        if col not in temp_df.columns:
            temp_df[col] = np.nan
    temp_df = temp_df[COLUMNAS]
    df_list.append(temp_df)

# Concatena todos los dataframes
df = pd.concat(df_list, ignore_index=True)

df.to_csv("var_full.csv", sep=";", index=False)

print(df.head())
