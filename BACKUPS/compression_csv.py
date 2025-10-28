import csv
import json

# Load drug list from the ATC JSON file
with open("ATC_farmaco_ENG.json", "r") as f:
    atc_data = json.load(f)
drug_list = [str(list(entry.values())[0]) for entry in atc_data]

# Build a mapping from drug name to list of genes
drug_to_genes = {drug: [] for drug in drug_list}

with open("relationships_trimmed.csv", "r", encoding="utf-8") as csvfile:
    csvreader = csv.reader(csvfile, delimiter=";")
    next(csvreader, None)  # Skip header if present
    for row in csvreader:
        drug, gene = row[0], row[1]
        if drug in drug_to_genes:
            drug_to_genes[drug].append(gene)

# Convert to desired output format: a list of dicts, each with the drug as key and gene list as value
output = [{drug: genes} for drug, genes in drug_to_genes.items() if genes]

# Write to output JSON
with open("drug_gene_output.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
