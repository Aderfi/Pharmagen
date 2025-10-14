import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json
import glob
import src.config.config as cfg
from src.config.config import *
from .metrics import *
from pathlib import Path
import pandas as pd
import numpy as np

class PGenInputDataset:
    def __init__(self):
        self.data = pd.DataFrame()
        self.encoders = {}
        self.cols = []

    def fit_encoders(self, df):
        # Inicializa y ajusta los encoders para cada columna categórica
        for col in df.columns:
            self.encoders[col] = LabelEncoder()
            self.encoders[col].fit(df[col].astype(str))

    def transform(self, df):
        # Aplica los encoders ajustados
        for col in df.columns:
            df[col] = self.encoders[col].transform(df[col].astype(str))
        return df

    def load_data(self, csv_files, cols):
        self.cols = cols
        df = pd.concat([pd.read_csv(f, sep=';', usecols=cols, index_col=False, dtype=str) for f in csv_files], ignore_index=True)
        # Crear la columna combinada Drug_Geno
        df["Drug_Geno"] = df["Drug"].astype(str) + "_" + df["Genotype"].astype(str)
        # Reordenar columnas para que Drug_Geno esté primero
        ordered_cols = ["Drug_Geno"] + [col for col in df.columns if col not in ["Drug_Geno"]]
        df = df[ordered_cols]

        df_cache = df.copy()
        if 'Drug' in df_cache: df_cache.pop('Drug')
        if 'Genotype' in df_cache: df_cache.pop('Genotype')
        df["stratify_col"] = df_cache.apply(lambda x: "_".join(x.astype(str)), axis=1)

        with open(Path(f"{PGEN_MODEL_DIR}/train_data/json_dicts/geno_alleles_dict.json"), "r") as f:
            equivalencias = json.load(f)
        if 'Genotype' in df:
            df['Genotype'] = df['Genotype'].map(lambda x: equivalencias.get(x, x))

        # Ajusta los encoders UNA VEZ con TODO el dataset
        self.fit_encoders(df)
        df = self.transform(df)

        counts = df['stratify_col'].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df['stratify_col'].isin(suficientes)]
        self.data = df.drop(columns=['stratify_col']).reset_index(drop=True)
        return self.data

    def get_tensors(self, cols):
        tensors = {}
        for col in cols:
            tensors[col.lower()] = torch.tensor(self.data[col].values, dtype=torch.long)
        return tensors

class PGenDataset(Dataset):
    def __init__(self, df):
        self.tensors = {col.lower(): torch.tensor(df[col].values, dtype=torch.long)
                        for col in df.columns}

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}
    
def train_data_load(targets):
    csv_path = MODEL_TRAIN_DATA
    # Asegurarse de que Drug y Genotype estén presentes para formar Drug_Geno
    read_cols = list(set(targets) | {"Drug", "Genotype"})
    csvfiles = glob.glob(f"{csv_path}/*.csv") 
    df = pd.concat([pd.read_csv(f, sep=';', dtype=str, usecols=read_cols) for f in csvfiles], ignore_index=True)
    # Añadir Drug_Geno para consistencia
    df["Drug_Geno"] = df["Drug"].astype(str) + "_" + df["Genotype"].astype(str)
    cols = ["Drug_Geno"] + [col for col in df.columns if col not in ["Drug_Geno"]]
    
    with open(f"{csv_path}/json_dicts/geno_alleles_dict.json", "r") as f:
        equivalencias = json.load(f)
    return csvfiles, cols, df, equivalencias


def explode_drug_genotype(df, drug_col='Drug', genotype_col='Genotype', drug_sep=',', genotype_sep=','):
    # Elimina espacios extra y divide por los separadores
    df = df.copy()
    df[drug_col] = df[drug_col].fillna('').astype(str)
    df[genotype_col] = df[genotype_col].fillna('').astype(str)
    df[drug_col] = df[drug_col].str.replace(' / ', '/').str.replace(' ,', ',').str.replace(', ', ',')
    df[genotype_col] = df[genotype_col].str.replace(' / ', '/').str.replace(' ,', ',').str.replace(', ', ',')

    # Divide y explota ambas columnas
    df[drug_col] = df[drug_col].str.split(drug_sep)
    df[genotype_col] = df[genotype_col].str.split(genotype_sep)
    df = df.explode(drug_col)
    df = df.explode(genotype_col)
    df[drug_col] = df[drug_col].str.strip()
    df[genotype_col] = df[genotype_col].str.strip()
    # Añadir columna Drug_Geno combinada para consistencia
    df["Drug_Geno"] = df[drug_col].astype(str) + "_" + df[genotype_col].astype(str)
    return df

def split_no_leakage(df, drug_col='Drug', genotype_col='Genotype', frac=0.8, seed=27):
    # Explota las combinaciones
    df_exploded = explode_drug_genotype(df, drug_col, genotype_col)
    # Crea columna combinada única
    df_exploded['Drug_Genotype'] = df_exploded[drug_col] + '_' + df_exploded[genotype_col]
    unique_combos = df_exploded['Drug_Genotype'].unique()
    np.random.seed(seed)
    train_combos = np.random.choice(unique_combos, int(frac * len(unique_combos)), replace=False)
    val_combos = list(set(unique_combos) - set(train_combos))
    train_df = df_exploded[df_exploded['Drug_Genotype'].isin(train_combos)].reset_index(drop=True)
    val_df = df_exploded[df_exploded['Drug_Genotype'].isin(val_combos)].reset_index(drop=True)
    return train_df, val_df
