import json

import numpy as np
import pandas as pd
from rapidfuzz import process


def normaliza(text, dict=None):
    if pd.isnull(text):
        return ""
    text = text.upper()
    text = text.replace("DRUGS", "").replace("DRUG", "")
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = text.strip()
    return text


df = pd.read_csv("var_full.csv", sep=";", dtype=str)

atc_dict = pd.read_json("ATC_drug_dict_ENG.json", typ="series").to_dict()

map_dict_norm = {normaliza(v): k for k, v in atc_dict.items()}
atc_dict = {v: k for k, v in atc_dict.items()}


df.insert(1, "ATC", df["Drugs"].map(atc_dict))


for index, row in df[df["ATC"].isna()].iterrows():
    if "/" in str(row["Drugs"]):
        drugs = [drug.strip() for drug in row["Drugs"].split("/")]
        atc_codes = [
            atc_dict.get(drug) for drug in drugs if atc_dict.get(drug) is not None
        ]
        if atc_codes:
            df.at[index, "ATC"] = ",".join(atc_codes)  # type: ignore
    if "," in str(row["Drugs"]):
        drugs = [drug.strip() for drug in row["Drugs"].split(",")]
        atc_codes = [
            atc_dict.get(drug) for drug in drugs if atc_dict.get(drug) is not None
        ]
        if atc_codes:
            df.at[index, "ATC"] = ",".join(atc_codes)  # type: ignore


def fuzzy_map(val, score_cutoff=80):
    choices = list(atc_dict.keys())
    if pd.isnull(val):
        return np.nan
    match, score, _ = process.extractOne(val, choices)
    return match if score >= score_cutoff else np.nan


for index, row in df[df["ATC"].isna()].iterrows():
    drug = row["Drugs"]
    fuzzy_result = fuzzy_map(drug)
    if pd.notnull(fuzzy_result):
        df.at[index, "ATC"] = map_dict_norm.get(fuzzy_result)

print(df["ATC"].isna().sum())

df.to_csv("test.csv", sep=";", index=False)
