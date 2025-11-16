import pandas as pd
from tabulate import tabulate
import numpy as np

df = pd.read_csv('snp_summary.tsv', sep='\t', dtype=str)

df[['Ref_Allele', 'Alt_Allele']] = df['variant'].str.split('>', expand=True)


df.to_csv('snp_summary.tsv', sep='\t', index=False)
print(df.sample(5))
print(df.columns)