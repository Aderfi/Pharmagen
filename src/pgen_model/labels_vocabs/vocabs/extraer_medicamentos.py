import json
import pandas as pd
import numpy as np

# Carga el archivo JSON
with open("drug_gene.json", "r", encoding="utf-8") as f:
    drug_gene_dict = json.load(f)

genes_list = []
drug_list = []

for drug, genes in drug_gene_dict.items():
    drug_list.append(drug)
    for gene in genes:
        genes_list.append(gene)

with open("genes_list.txt", "w", encoding="utf-8") as f_gene, open(
    "drug_list.txt", "w", encoding="utf-8"
) as f_drug:
    for drug in drug_list:
        f_drug.write(f"{drug}\n")
    for gene in genes_list:
        f_gene.write(f"{gene}\n")
