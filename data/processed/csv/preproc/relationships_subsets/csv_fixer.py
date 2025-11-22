# drugs.csv: "PharmGKB Accession Id","Name","Generic Names","Trade Names","Brand Mixtures","Type","Cross-references","SMILES","InChI","Dosing Guideline","External Vocabulary","Clinical Annotation Count","Variant Annotation Count","Pathway Count","VIP Count","Dosing Guideline Sources","Top Clinical Annotation Level","Top FDA Label Testing Level","Top Any Drug Label Testing Level","Label Has Dosing Info","RxNorm Identifiers","ATC Identifiers","PubChem Compound Identifiers","Top CPIC Pairs Level","FDA Label has Prescribing Info","In FDA PGx Association Sections"
# relationships.csv: "Entity1_id","Entity1_name","Entity1_type","Entity2_id","Entity2_name","Entity2_type","Evidence","Association","PK","PD","PMIDs"
# study_parameters.csv:"Study Parameters ID","Variant Annotation ID","Study Type","Study Cases","Study Controls","Characteristics","Characteristics Type","Frequency In Cases","Allele Of Frequency In Cases","Frequency In Controls","Allele Of Frequency In Controls","P Value","Ratio Stat Type","Ratio Stat","Confidence Interval Start","Confidence Interval Stop","Biogeographical Groups"


# var_drug_ann.csv: "Variant Annotation ID","Variant/Haplotypes","Gene","Drug(s)","PMID","Phenotype Category","Significance","Notes","Sentence","Alleles","Specialty Population","Metabolizer types","isPlural","Is/Is Not associated","Direction of effect","PD/PK terms","Multiple drugs And/or","Population types","Population Phenotypes or diseases","Multiple phenotypes or diseases And/or","Comparison Allele(s) or Genotype(s)","Comparison Metabolizer types"
# var_fa_ann.csv: "Variant Annotation ID","Variant/Haplotypes","Gene","Drug(s)","PMID","Phenotype Category","Significance","Notes","Sentence","Alleles","Specialty Population","Assay type","Metabolizer types","isPlural","Is/Is Not associated","Direction of effect","Functional terms","Gene/gene product","When treated with/exposed to/when assayed with","Multiple drugs And/or","Cell type","Comparison Allele(s) or Genotype(s)","Comparison Metabolizer types"
# var_pheno_ann.csv: "Variant Annotation ID","Variant/Haplotypes","Gene","Drug(s)","PMID","Phenotype Category","Significance","Notes","Sentence","Alleles","Specialty Population","Metabolizer types","isPlural","Is/Is Not associated","Direction of effect","Side effect/efficacy/other","Phenotype","Multiple phenotypes And/or","When treated with/exposed to/when assayed with","Multiple drugs And/or","Population types","Population Phenotypes or diseases","Multiple phenotypes or diseases And/or","Comparison Allele(s) or Genotype(s)","Comparison Metabolizer types"

import glob
import re

import pandas as pd

csv_files = glob.glob("*.csv")
match_list = {}

for file in csv_files:
    # Read the CSV file
    df = pd.read_csv(file, sep=";", dtype=str)
    """
    filename = file.split('.')[0].split('_', maxsplit=1)[1]
    col1_name, col2_name = filename.split('_')

    df.rename(columns={df.columns[0]: f"{col1_name}_ID", df.columns[1]: f"{col1_name}", df.columns[3]: f"{col2_name}_ID", df.columns[4]: f"{col2_name}"}, inplace=True)
    df['LADME'] = pd.concat([df['PD'], df['PK']], axis=0).reset_index(drop=True)
    df.drop(columns=['PD', 'PK', 'PMIDs', 'Evidence'], inplace=True)
    """

    pattern = "1453073100"
    try:
        for value in df["Variant_ID"]:
            match = re.search(pattern, str(value))
            if match:
                match_list[file] = True
            continue
    except (KeyError, TypeError, pd.errors.ParserError):
        continue

    print(match_list)
