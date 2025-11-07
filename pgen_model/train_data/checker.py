import pandas as pd
import re
import json
import numpy as np
import itertools
import sys
from tabulate import tabulate  
import itertools 

df = pd.read_csv('final_test_genalle.tsv', sep='\t')

cols = ['Drug', 'Variant/Haplotypes', 'Genalle', 'Gene', 'Allele', 'Phenotype_outcome', 'Effect_direction', 'Effect_type']

print(tabulate(df[cols].sample(50), headers='keys', tablefmt='psql', showindex=False, maxcolwidths=20))

