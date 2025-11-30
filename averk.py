import pandas as pd
from rapidfuzz import process, fuzz
import re
from tqdm import tqdm

df_dict = pd.read_csv("Drug_Compount_processed.tsv", sep="\t", usecols=["Compound_CID", "Name", "Synonyms"])

dict_comp = {}
for row in df_dict.itertuples(index=False):
    synonims_list = re.split(r'[|,\/]', row.Synonyms) if pd.notna(row.Synonyms) else []
    dict_comp[row.Name] = row.Compound_CID
    for synonym in synonims_list:
        dict_comp[synonym] = row.Compound_CID
print(f"Total unique compound names and synonyms: {len(dict_comp)}")

tqdm.pandas()
def mapeo_chuli(x):
    try:
        query_list = re.split(r'[|,\/]', x) if re.match(r'[|,\/]', x) else [x]

        mejor_match = process.extractOne(
            query=' '.join(query_list),
            choices=dict_comp.keys(),
            scorer=fuzz.token_set_ratio,
            score_cutoff=95
        )
        if mejor_match:
            return dict_comp[mejor_match[0]]
        else:
            segunda_vuelta = process.extractOne(
                query=' '.join(query_list),
                choices=dict_comp.keys(),
                scorer=fuzz.token_set_ratio,
                score_cutoff=90
            )
            if segunda_vuelta:
                return dict_comp[segunda_vuelta[0]]
        return "No_match"
    except KeyError as e:
        return "No_match"




data_df = pd.read_csv("train_data/final_enriched_data.tsv", sep="\t")

data_df["Drug_CID"] = data_df["Drug"].progress_map(mapeo_chuli)

data_df.to_csv("final_mapped_data.tsv", sep="\t", index=False)

