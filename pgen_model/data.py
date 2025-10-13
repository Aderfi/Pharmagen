import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json

class PGenInputDataset:
    def __init__(self):
        self.data = None
        self.encoders = {}

    def load_data(self, PMODEL_DIR, csv_files, cols):
        df = pd.concat([pd.read_csv(f, sep=';', usecols=cols, index_col=False, dtype=str) for f in csv_files], ignore_index=True)
        df_cache = df.copy()
        df_cache.pop('Drug')
        df_cache.pop('Genotype')
        df["stratify_col"] = df_cache.apply(lambda x: "_".join(x.astype(str)), axis=1)

        with open(f"{PMODEL_DIR}/train_data/json_dicts/geno_alleles_dict.json", "r") as f:
            equivalencias = json.load(f)
        df['Genotype'] = df['Genotype'].map(lambda x: equivalencias.get(x, x))

        for col in df.columns:
            self.encoders[col] = LabelEncoder()
            df[col] = self.encoders[col].fit_transform(df[col].astype(str))

        counts = df['stratify_col'].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df['stratify_col'].isin(suficientes)]
        self.data = df.drop(columns=['stratify_col']).reset_index(drop=True)
        return self.data

    def get_tensors(self):
        tensors = {}
        for col in ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']:
            tensors[col.lower()] = torch.tensor(self.data[col].values, dtype=torch.long)
        return tensors

class PGenDataset(Dataset):
    def __init__(self, df):
        self.tensors = {col.lower(): torch.tensor(df[col].values, dtype=torch.long)
                        for col in ['Drug', 'Genotype', 'Outcome', 'Variation', 'Effect', 'Entity']}

    def __len__(self):
        return len(self.tensors['drug'])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}