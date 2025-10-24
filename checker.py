import pandas as pd
import re
import json

df = pd.read_csv("train_therapeutic_outcome.csv", sep=';')

with open('geno_alleles_dict.json') as f:
    equivalencias = json.load(f)
    
equivalencias_cleaned = {}
for k, v in equivalencias.items():
    if re.match(r'rs[0-9]+', k):
        equivalencias_cleaned[k] = v
    else:
        continue

df['Genotype'] = df['Genotype'].map(equivalencias_cleaned, na_action='ignore')

df.to_csv("train_therapeutic_outcome_cleaned.csv", sep=';', index=False)