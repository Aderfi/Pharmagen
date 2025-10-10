import csv
import json 
import os 
import sys
import pandas as pd
import re

with open('ATC_drug_dict_ENG.json', 'r', encoding='utf-8') as f:
    drug_gene_dict = json.load(f)

drug_gene_dict = list((v, k) for k, v in drug_gene_dict.items())

with open('var_fa_ann_model.csv', 'r', encoding='utf-8') as f:
    var_drug_ann = pd.read_csv(f, sep=';', dtype=str, index_col=False)
    

print(var_drug_ann.columns)
print(type(var_drug_ann))
print(type(drug_gene_dict))
print(drug_gene_dict[:10])



for k, row in var_drug_ann.iterrows():
    
    drug_list = []
    cod_atc_list = []
    cod_atc_all = []
        
    drug = row['Drug'].strip().upper()

    cod_atc_all = [cod_atc for drug_name, cod_atc in drug_gene_dict if drug_name == drug]
    

    if "," in row['Drug'] or "/" in row['Drug']:
        match = re.search(r"[,/]", str(row['Drug']))
        if match:
            for i in re.split(r"[,/]", str(row['Drug'])):
                drug = i.strip().upper()
                cod_atc_all.extend([cod_atc for drug_name, cod_atc in drug_gene_dict if drug_name == drug])
                

    var_drug_ann.at[k, 'ATC'] = ",".join(cod_atc_all)


with open('var_fa_ann_model_with_ATC.csv', 'w', encoding='utf-8', newline='') as f:
    var_drug_ann.to_csv(f, sep=';', index=False)