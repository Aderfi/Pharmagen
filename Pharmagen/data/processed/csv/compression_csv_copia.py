import json
import pandas as pd

# Cargar los datos ATC
with open('ATC_farmaco_ENG.json', 'r', encoding='utf-8') as drug_info:
    atc_data = json.load(drug_info)

# Leer el CSV con una codificación más permisiva
try:
    csv_df = pd.read_csv('relationships_trimmed.csv', delimiter=';', encoding='utf-8')
except UnicodeDecodeError:
    csv_df = pd.read_csv('relationships_trimmed.csv', delimiter=';', encoding='latin1')

# Construir la lista de nombres de medicamentos desde el JSON
atc_drugs_name = []
for entry in atc_data:
    # entry es dict, obtenemos todos los valores de ese dict
    for v in entry.values():
        atc_drugs_name.append(v)

# Añadir nombres de medicamentos que aparecen en el CSV y no están en la lista
for drug_name in csv_df['Name']:
    if drug_name not in atc_drugs_name:
        atc_drugs_name.append(drug_name)

relationships_list = []

# Para cada medicamento, buscar los genes asociados en el CSV
for drug in atc_drugs_name:
    genes_list = []
    for idx, row in csv_df.iterrows():
        if row['Name'] == drug:
            gene = row['Mutation']
            if gene not in genes_list:
                genes_list.append(gene)
    relationships_list.append({drug: genes_list})

# Guardar el resultado en JSON
with open('relationships_long.json', 'w', encoding='utf-8') as json_output:
    json.dump(relationships_list, json_output, ensure_ascii=False, indent=4)

# Guardar el resultado en CSV (opcional, como tabla expandida)
csv_expanded = []
for rel in relationships_list:
    for drug, genes in rel.items():
        for gene in genes:
            csv_expanded.append({'Drug': drug, 'Gene': gene})

csv_df_out = pd.DataFrame(csv_expanded)
csv_df_out.to_csv('relationships_long.csv', index=False, encoding='utf-8')

print('\n\t----------Finiquitao-------------------\n')