import pandas as pd
import re
import json
import numpy as np
import itertools
import sys
from tabulate import tabulate  
import itertools 

df = pd.read_csv('relationships_associated_corrected.tsv', sep='\t', dtype=str)

mask = df['Entity1_name'].str.contains(r'rs\d+', na=False) | df['Entity2_name'].str.contains(r'rs\d+', na=False)


list_rsid = [i for i in df['Entity1_name'].tolist() if re.match(r'rs\d+', str(i))] + \
            [i for i in df['Entity2_name'].tolist() if re.match(r'rs\d+', str(i))]

dict_rsid = {k:'Pend' for k in list_rsid}

print(dict_rsid)
print(f"Total unique rsIDs found: {len(dict_rsid)}")

with open('rsid_dict.json', 'w') as f:
    json.dump(dict_rsid, f, indent=1)





