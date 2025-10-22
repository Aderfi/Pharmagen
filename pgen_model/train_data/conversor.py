import pandas as pd

df = pd.read_csv('train_therapeutic_outcome.csv', sep=';')

# Obtener el número de clases únicas para cada variable categórica
n_drugs = df['Drug'].nunique()
n_genes = df['Gene'].nunique()
n_alleles = df['Allele'].nunique()
n_genotypes = df['Genotype'].nunique()

print(f"Número de clases únicas: {n_drugs}, {n_genes}, {n_alleles}, {n_genotypes}")


       