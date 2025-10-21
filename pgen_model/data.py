import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import json
import glob
from pathlib import Path
import numpy as np
import src.config.config as cfg
from src.config.config import *
from .metrics import *
import sys
import re
import random

def load_equivalencias(csv_path):
    with open(f"{csv_path}/json_dicts/geno_alleles_dict.json", "r") as f:
        equivalencias = json.load(f)
        
        for k, v in equivalencias.items():
            if re.match(r'rs[0-9]+', k):
                continue
            else:
                equivalencias[k] = np.nan

        equivalencias = equivalencias.pop(key) if (key := 'nan') in equivalencias else equivalencias
        
    return equivalencias

class PGenInputDataset:
    """
    Clase para estructurar los datos de entrada para multisalida.
    """
    def __init__(self):
        self.data = pd.DataFrame()
        self.encoders = {}
        self.cols = []
        self.target_cols = []

    def fit_encoders(self, df, target_cols):
        for col in df.columns:
            if col in target_cols:
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str))
            elif col != "stratify_col":
                self.encoders[col] = LabelEncoder()
                self.encoders[col].fit(df[col].astype(str))

    def transform(self, df, target_cols):
        for col in df.columns:
            if col in target_cols:
                df[col] = self.encoders[col].transform(df[col].astype(str))
            elif col in self.encoders and col != "stratify_col":
                df[col] = self.encoders[col].transform(df[col].astype(str))
        return df

    def load_data(self, csv_files, cols, targets, equivalencias):
        self.cols = [col.lower() for col in cols]
        self.target_cols = [t.lower() for t in targets]
        '''
        if isinstance(csv_files, (list, tuple)):
            df = pd.concat([pd.read_csv(f, sep=';', index_col=False) for f in csv_files], ignore_index=True)
        else:'''
        csv_files = Path(MODEL_TRAIN_DATA / 'train_therapeutic_outcome.csv')
        df = pd.read_csv(str(csv_files), sep=';', index_col=False)
        df.columns = [col.lower() for col in df.columns]
        df = df[[c.lower() for c in cols]]
        targets = [t.lower() for t in targets]

        # Normaliza las columnas target a string
        for t in targets:
            df[t] = df[t].astype(str)
        
        
        
        # stratify_col sigue combinada para splits, pero targets por separado
        df['stratify_col'] = df[targets].astype(str).agg("_".join, axis=1)
        df = df.dropna(subset=targets, axis=0, ignore_index=True)
        self.fit_encoders(df, targets)
        df = self.transform(df, targets)
        counts = df['stratify_col'].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df['stratify_col'].isin(suficientes)]
        self.data = df.drop(columns=['stratify_col']).reset_index(drop=True)
        
        
        df = df.dropna(subset=targets, axis=0, ignore_index=True)
        self.fit_encoders(df, targets)
        df = self.transform(df, targets)
        self.data = df.reset_index(drop=True)
        
        return self.data
    
    def get_tensors(self, cols):
        tensors = {}
        for col in cols:
            if col in self.data:
                tensors[col.lower()] = torch.tensor(self.data[col].values, dtype=torch.long)
        for target in self.target_cols:
            if target in self.data:
                tensors[target] = torch.tensor(self.data[target].values, dtype=torch.long)
        return tensors

class PGenDataset(Dataset):
    def __init__(self, df, target_cols):
        self.tensors = {col.lower(): torch.tensor(df[col].values, dtype=torch.long)
                        for col in df.columns if col != 'stratify_col'}
        self.target_cols = [t.lower() for t in target_cols]

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.tensors.items()}

def train_data_load(targets):
    csv_path = MODEL_TRAIN_DATA
    targets = [t.lower() for t in targets]
    read_cols = list(set(targets) | {"Drug", "Gene", "Allele", "Genotype"})
    read_cols = [c.lower() for c in read_cols]
    csvfiles = glob.glob("*.csv") #glob.glob(f"{csv_path}/*.csv")
    equivalencias = load_equivalencias(csv_path)
    return csvfiles, read_cols, equivalencias