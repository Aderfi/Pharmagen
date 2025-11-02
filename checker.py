import pandas as pd
import re
import json
import numpy as np
import itertools
import sys

file="test_var_nofa_mapped_filled_with_effect_phenotype_ids.tsv"


df = pd.read_csv(file, sep="\t", dtype=str, )

'''
Col: ATC, NaN Count: 1217, NaN Percent: 9.17%
Col: Drugs, NaN Count: 345, NaN Percent: 2.60%n    
Col: Variant/Haplotypes, NaN Count: 0, NaN Percent: 0.00%
Col: Gene, NaN Count: 371, NaN Percent: 2.80%
Col: Alleles, NaN Count: 406, NaN Percent: 3.06%
Col: Phenotype_outcome, NaN Count: 82, NaN Percent: 0.62%
Col: Effect_direction, NaN Count: 498, NaN Percent: 3.75%
Col: Effect_type, NaN Count: 3, NaN Percent: 0.02%
Col: Effect_phenotype, NaN Count: 6215, NaN Percent: 46.83%
Col: Metabolizer types, NaN Count: 12627, NaN Percent: 95.15%
Col: Population types, NaN Count: 2585, NaN Percent: 19.48%
Col: Population Phenotypes or diseases, NaN Count: 3794, NaN Percent: 28.59%
Col: Comparison Allele(s) or Genotype(s), NaN Count: 1769, NaN Percent: 13.33%
Col: Comparison Metabolizer types, NaN Count: 12713, NaN Percent: 95.80%
Col: Notes, NaN Count: 2001, NaN Percent: 15.08%
Col: Sentence, NaN Count: 0, NaN Percent: 0.00%
Col: Variant Annotation ID, NaN Count: 0, NaN Percent: 0.00%
'''
'''
df = df.dropna(subset=['Phenotype_outcome']).reset_index(drop=True)
df = df.dropna(subset=['Effect_type']).reset_index(drop=True)


df['ATC'] = df['ATC'].fillna('ATC_Desconocido')
df['Drugs'] = df['Drugs'].fillna('Farmaco_Desconocido/NoRelacionado')
df['Variant/Haplotypes'] = df['Variant/Haplotypes'].fillna('No_Variant')
df['Gene'] = df['Gene'].fillna('Intron/No_Especificado')
df['Alleles'] = df['Alleles'].fillna('Alelle_NoEspecificado')
df['Phenotype_outcome'] = df['Phenotype_outcome'].fillna('No_Phenotype')
df['Effect_direction'] = df['Effect_direction'].fillna('Indeterminado')
df['Effect_type'] = df['Effect_type'].fillna('No_Effect_type')
df['Effect_phenotype'] = df['Effect_phenotype'].fillna('Fenotipo_No_Especificado')

df.to_csv("var_nofa_mapped_filled.tsv", sep="\t", index=False)


df.rename(columns={
    'ATC': 'ATC',
    'Drugs': 'Drug',
    'Variant/Haplotypes': 'Variant/Haplotypes',
    'Gene': 'Gene',
    'Alleles': 'Alleles',
    'Phenotype_outcome': 'Phenotype_outcome',
    'Effect_direction': 'Effect_direction',
    'Effect_type': 'Effect_type',
    'Effect_phenotype': 'Effect_phenotype',
    'Metabolizer types': 'Metabolizer types',
    'Population types': 'Population types',
    'Population Phenotypes or diseases': 'Pop_Phenotypes/Diseases',
    'Comparison Allele(s) or Genotype(s)': 'Comparison Allele(s) or Genotype(s)',
    'Comparison Metabolizer types': 'Comparison Metabolizer types',
    'Notes': 'Notes',
    'Sentence': 'Sentence',
    'Variant Annotation ID': 'Variant Annotation ID'
}, inplace=True)
'''

print(df)

lista_rs = []

pattern = r'^rs[0-9]+'

for row in df['Variant/Haplotypes'].values:
    if re.match(pattern, str(row)):
        lista_rs.append(row)
    else:
        continue

print("Total rsIDs found:", len(lista_rs))
with open("rsIDs_extracted.txt", "w") as f:
    for rsid in lista_rs:
        f.write(rsid + "\n")