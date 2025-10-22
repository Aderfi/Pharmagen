import pandas as pd
import glob
import re
import json
from pathlib import Path
import numpy as np
'''
targets = ['Outcome', 'Variation']

columns_unique_classes = {}
outcome_unique_classes = {}
variation_unique_classes = {}

outcome_unique_classes = {k: v for k, v in df['Outcome'].value_counts().items()}
variation_unique_classes = {k: v for k, v in df['Variation'].value_counts().items()}



outcome_unique_classes = dict(sorted(outcome_unique_classes.items(), key=lambda x: x[1]))
variation_unique_classes = dict(sorted(variation_unique_classes.items(), key=lambda x: x[1]))



with open('outcome_unique_classes.json', 'w') as f, open('variation_unique_classes.json', 'w') as g:
    json.dump(outcome_unique_classes, f, indent=4)
    f.write('\n')
    json.dump(variation_unique_classes, g, indent=4)
    g.write('\n')
    
import pandas as pd

unique_dict = {}

# Contar las ocurrencias de cada combinación única
for key in df.groupby(['Outcome', 'Variation']).size().index:
    unique_dict[key] = len(df[(df['Outcome'] == key[0]) & (df['Variation'] == key[1])])

# Ordenar por cantidad (de menor a mayor) y escribir en archivo
with open('unique_classes_num.txt', 'w') as h:
    for k, v in sorted(unique_dict.items(), key=lambda x: x[1]):
        h.write(f"{k}: {v}\n")

# Ordenar alfabéticamente y escribir en archivo
with open('unique_classes_summary_alph.txt', 'w') as j:
    for k, v in sorted(unique_dict.items(), key=lambda x: x[0]):
        j.write(f"{k}: {v}\n")
'''
'''
lista_unicos = df['Variation'].value_counts().to_dict()

for k, v in lista_unicos.items():
    print(f"'{k}': {v} \n")
    '''
    
import pandas as pd
import itertools

# Columnas objetivo para el producto cartesiano
TARGET_COLUMNS = ["Drug", "Gene","Genotype", "Allele" "Outcome", "Variation", "Effect", "Entity"]

def split_cell(cell):
    return [item.strip() for item in str(cell).split(",") if item.strip()]

def expand_row(row, columns):
    lists = [split_cell(row[col]) if col in row else [""] for col in columns]
    for combination in itertools.product(*lists):
        new_row = row.copy()
        for col, value in zip(columns, combination):
            new_row[col] = value
        yield new_row

def cartesian_csv_pandas(input_file, output_file):
    df = pd.read_csv(input_file, sep=";", dtype=str, on_bad_lines='warn')
    expanded_rows = []
    for _, row in df.iterrows():
        for expanded_row in expand_row(row, TARGET_COLUMNS):
            expanded_rows.append(expanded_row)
    new_df = pd.DataFrame(expanded_rows)
    new_df.to_csv(output_file, sep=";", index=False, encoding='utf-8')

# Ejemplo de uso:
cartesian_csv_pandas("Libro1.csv", "expanded_final_data_model_outcome_clean.csv")