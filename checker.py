import pandas as pd
import re
import json
import numpy as np
import itertools
import sys
from tabulate import tabulate  
import itertools 

df = pd.read_csv('snp_genes_table.tsv', sep='\t', dtype=str)
'''
df['Acc'] = df['Acc'].fillna('')
df['Gene_Name'] = df['Gene_Name'].fillna('INTRON')

df['Acc'] = df['Acc'].replace(r'NC_[0-9]+.[0-9]+:+[0-9:]+', '', regex=True)
                                #NC_000006.12:31815977

df['Genalle'] = str('') # Nueva columna para guardar Gene_Acc
df['Genalle_Test'] = str('')  # Nueva columna para guardar Acc sin prefijo

def generar_genalle(row):
    raw_acc = str(row['Acc']).strip()
    raw_gene = str(row['Gene_Name']).strip()
    acc_list = [x.strip() for x in raw_acc.split(',') if x.strip()]
    temp_genes = [x.strip() for x in raw_gene.split(',') if x.strip()]
    gene_list = [g for g in temp_genes]
                  
    if not acc_list: acc_list = [raw_acc]
    if not gene_list: gene_list = [raw_gene]
    
    acc_list = [str(x).split(':')[1] for x in acc_list if ':' in str(x)]
    
    combinaciones = [f"{g}_{a}" for g, a in itertools.product(gene_list, acc_list)]
    
    return ','.join(combinaciones)

df['Genalle'] = df.apply(generar_genalle, axis=1)        

def arreglar_acc(row):
    raw_allele = str(row['Allele']).strip()
    raw_gene = str(row['Gene_Name']).strip()
    gene_list = [x.strip() for x in raw_gene.split(',')] if ',' in raw_gene else [raw_gene]
    
    if not raw_allele or raw_allele == 'nan':
        return ''
    combined = [f"{g}_{raw_allele}" for g in gene_list]
    
    return ','.join(combined)
    

print(tabulate(df[['Acc', 'Gene_Name', 'Genalle', 'Genalle_Test']].sample(10), headers='keys', tablefmt='psql') if 'tabulate' in sys.modules else df[['Acc', 'Gene_Name', 'Genalle', 'Genalle_Test']].sample(10))
'''
#Id	SNP_Id	Acc	Allele	SNP_type	Chr_Position	Gene_Name	Genalle	Genalle_Test


df['Check'] = df['Id'] == df['SNP_Id']

# Filtramos (el signo ~ invierte la selección, es decir, busca los False)
mask = df[~df['Check']]

print(mask[['Id', 'SNP_Id']].sample(10))