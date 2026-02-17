import json

import pandas as pd
import tomlkit

with open('tomli_ATC_drug_med.toml', 'rb') as f:
    dictio = tomlkit.load(f)


dict_json = dict(dictio)

with open('drug_list.txt', 'w') as f:
    for _k, v in dict_json.items():
        f.write(v + '\n')
