import pandas as pd
import re
import json
import numpy as np
import itertools
import sys
from tabulate import tabulate   

df = pd.read_csv('final_test_filled.tsv', sep='\t')


mask = df['Effect_type'].value_counts() < 20

print("Clases con menos de 20 muestras en 'Effect_type':")
print(mask[mask].index.tolist())
