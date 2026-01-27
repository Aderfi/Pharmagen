import pandas as pd
import re
from rapidfuzz import process, fuzz

with open('data/dicts/ATC_drug_med.json', 'r', encoding='utf-8') as file:
    JSON_DICT = pd.read_json(file, typ='series').to_dict()
atc_dict = {v[0]: [k] for k, v in JSON_DICT.items()}

for k, v in atc_dict.items():
    atc_dict[b[0]] = a

def translator():
    atc_codes = None



df = pd.DataFrame(columns=['ATC', 'farmaco'])

with open('conj_auton_expanded.txt', 'r', encoding='utf-8') as file:
    df["farmaco"] = [line.strip() for line in file.readlines()]

with open('data/dicts/ATC_drug_med.json', 'r', encoding='utf-8') as file:
    json_dict = pd.read_json(file, typ='series').to_dict()

atc_dict = {v[0]: [k] for k, v in json_dict.items()}

drug_dict_list = [v[0] for k, v in json_dict.items()]

df["ATC"] = df["farmaco"].map(translator)

print(df.sample(20))