import json
import pandas as pd
import glob
import re
import itertools
import csv
'''
csv_files = glob.glob("*.csv")
cols = ['Drug', 'Gene', 'Genotype', 'Alleles', 'Outcome', 'Variation', 'Effect', 'Entity']

df = pd.concat([pd.read_csv(f, sep=';', index_col=False, usecols=cols) for f in csv_files], ignore_index=True)

df.insert(8, 'Merged', '')

pattern = r"^[A-Z0-9]+ (poor|slow|ultrarapid|rapid|intermediate) (metabolizer|acetylator)$"
list_fix = []

for idx in df[df['Genotype'].str.match(pattern).tolist()].index:
    list_fix.append((idx, df.loc[idx, 'Genotype']))
    
for idx, geno in list_fix:
    df.loc[idx, 'Genotype'] = geno.split(' ')[0]
    df.loc[idx, 'Alleles'] = geno.split(' ')[1] + '_' + geno.split(' ')[2]

df['Variation'] = df['Variation'].str.replace('-', '_', regex=False)
df['Outcome'] = df['Outcome'].str.upper().str.replace('-', '_', regex=False)
df['Entity'] = df['Entity'].str.strip().str.replace(r'\s', '_', regex=False)


# Merge Effect and Entity into Merged, separated by a ':' if both exist
df['Merged'] = df.apply(lambda row: f"{row['Effect']}:{row['Entity']}" if pd.notna(row['Effect']) and pd.notna(row['Entity']) else (row['Effect'] if pd.notna(row['Effect']) else (row['Entity'] if pd.notna(row['Entity']) else None)), axis=1)
df.drop(columns=['Entity'], inplace=True)
df.drop(columns=['Effect'], inplace=True)

df['Merged'] = df['Merged'].str.strip()
df['Merged'] = df['Merged'].str.replace(r'\s', '_', regex=True)
df['Merged'] = df['Merged'].str.replace(r'__+', '_', regex=True)

df.rename(columns={'Merged': 'Effect'}, inplace=True)

df.to_csv('pre_concat.csv', sep=';', index=False)


# Leer el archivo de entrada
df = pd.read_csv("pre_concat.csv", sep=';', usecols=['Drug', 'Gene', 'Alleles', 'Genotype', 'Outcome', 'Variation', 'Effect'])

def expand_row(row):
    drugs = [d.strip() for d in re.split('[,/]', str(row['Drug']))]
    genes = [g.strip() for g in str(row['Gene']).split(',')]
    outcomes = [o.strip() for o in str(row['Outcome']).split(',')]
    # Estas columnas no se expanden, se copian igual en cada combinación
    alleles = row['Alleles']
    genotype = row['Genotype']
    effect = row['Effect']
    # Variations: solo reemplazar - por _
    variation = str(row['Variation']).rstrip('_') if re.search(r'[_]$', str(row['Variation'])) else str(row['Variation'])
    # Producto cartesiano de Drug, Gene, Outcome
    for drug, gene, outcome in itertools.product(drugs, genes, outcomes):
        yield {
            'Drug': drug,
            'Gene': gene,
            'Allele': alleles,
            'Genotype': genotype,
            'Outcome': outcome,
            'Variation': variation,
            'Effect': effect
        }

output_file = "expanded_output.csv"
final_columns = ['Drug', 'Gene', 'Allele', 'Genotype', 'Outcome', 'Variation', 'Effect']

with open(output_file, "w", newline='', encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=final_columns, delimiter=';')
    writer.writeheader()
    for _, row in df.iterrows():
        for expanded in expand_row(row):
            writer.writerow(expanded)

print(f"Expansión completada y guardada en {output_file}")
'''

import pandas as pd

df = pd.read_csv("expanded_output.csv", sep=';')

for col in df.columns:
    df[col] = df[col].apply(lambda x: f"NO_{col.upper()}_RELATED" if pd.isna(x) else x)

df.to_csv('final_output.csv', sep=';', index=False)

