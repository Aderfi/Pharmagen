'''
import pandas as pd
import os
import numpy as np
import json
import glob

csv_files = glob.glob("*.csv")

df = pd.concat([pd.read_csv(f, sep=';', usecols=['Gene', 'Genotype', 'Alleles'], index_col=False, dtype=str) for f in csv_files], ignore_index=True)
df = df.astype(str)



geno_alleles_dict = {}

for index, row in df.iterrows():
    row['Genotype'] = str(row['Genotype'])
    if "," in str(row['Genotype']):
        genotype = [i.strip() for i in str(row['Genotype']).split(",") if i.strip()]
    else:
        genotype = [str(row['Genotype']).strip()]
    
    gene_allele = row['Gene'] + ["_" + row['Alleles'] if str(row['Alleles']) != 'nan' else ''][0]
    
    for geno in genotype:
        geno = geno.strip()
        
        geno_alleles_dict[geno] = ''
        geno_alleles_dict[geno] += gene_allele

geno_alleles_out = geno_alleles_dict.copy()

for i in geno_alleles_dict.items():
    if i[1] == 'nan' or i[1] == '':
        geno_alleles_out.pop(i[0])




with open("geno_alleles.json", "w") as f:
    json.dump(geno_alleles_out, f, indent=2)
'''

import pandas as pd
import numpy as np
import json

with open("genes_list.txt", "r") as f:
    genes = f.readlines()
    genes = [gene.strip() for gene in genes]
    
with open("drug_list.txt", "r") as f:
    drugs = f.readlines()
    drugs = [drug.strip() for drug in drugs]
    
drug_genes_random_list = []

for i in range(50):
    random_number_drug = np.random.randint(1, len(drugs)+1)
    random_number_gene = np.random.randint(1, len(genes)+1)
    
    random_drug = drugs[random_number_drug-1]
    random_gene = genes[random_number_gene-1]
    
    drug_genes_random_list.append((random_drug, random_gene))
    
    
with open("drug_genes_random_list.json", "w") as f:
    json.dump(drug_genes_random_list, f, indent=2)

