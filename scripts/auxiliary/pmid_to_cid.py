import requests
import time
import pandas as pd
import json
import numpy as np


df = pd.read_table('pmid_to_cid.tsv', sep='\t', header=None, names=['pmid', 'cid'])

df['parent_cid'] = df['cid'].apply(lambda x: str(x).split('.')[0])
