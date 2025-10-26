import json

import numpy as np
import pandas as pd

df = pd.DataFrame(columns=['Drug', 'Genotype'])

with open("drug_gene_output.json", "r") as f:
    drug_gene_dict = json.load(f)

i=0

for drug, genes in drug_gene_dict.items():
    while i<50:
        random_drug = np.random.choice(drug, size=np.random.randint(1, len(drug)+1), replace=False)
        for gene in genes:
            random_gene = np.random.choice(genes, size=np.random.randint(1, len(genes)+1), replace=False)
            df = pd.concat([df, pd.DataFrame({'Drug': [drug], 'Genotype': [gene]})], ignore_index=True)
        i += 1



print(df.head())