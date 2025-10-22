import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer # <-- AÑADIDO
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
    Maneja tanto encoders single-label (LabelEncoder) como multi-label (MultiLabelBinarizer).
    """
    def __init__(self):
        self.data = pd.DataFrame()
        self.encoders = {}
        self.cols = []
        self.target_cols = []
        self.multi_label_cols = set() # <-- AÑADIDO: para rastrear columnas multi-etiqueta

    def fit_encoders(self, df):
        """
        Ajusta el encoder apropiado (LabelEncoder o MultiLabelBinarizer)
        para cada columna de input y target.
        """
        for col in df.columns:
            if col == "stratify_col":
                continue
            
            if col in self.multi_label_cols:
                # Usar MultiLabelBinarizer para columnas multi-etiqueta
                encoder = MultiLabelBinarizer()
                encoder.fit(df[col])
                self.encoders[col] = encoder
            elif col in self.cols: # self.cols tiene todos los inputs + targets
                # Usar LabelEncoder para inputs y targets de etiqueta única
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
                self.encoders[col] = encoder

    def transform(self, df):
        """
        Transforma el DataFrame usando los encoders ajustados.
        - LabelEncoder -> Columna de enteros
        - MultiLabelBinarizer -> Columna de listas de arrays (vectores binarios)
        """
        for col, encoder in self.encoders.items():
            if col not in df.columns:
                continue
            
            if isinstance(encoder, MultiLabelBinarizer):
                # Transformar y almacenar el resultado (una lista de arrays)
                df[col] = list(encoder.transform(df[col]))
            else: # LabelEncoder
                # Transformar como antes
                df[col] = encoder.transform(df[col].astype(str))
        return df

    def load_data(self, csv_files, cols, targets, equivalencias, multi_label_targets: list):
        """
        Carga y preprocesa los datos.
        'multi_label_targets' es una lista de nombres de columnas (ej. ['outcome_category'])
        que deben ser tratadas como multi-etiqueta.
        """
        self.cols = [col.lower() for col in cols]
        self.target_cols = [t.lower() for t in targets]
        self.multi_label_cols = set(t.lower() for t in multi_label_targets or [])
        
        '''
        if isinstance(csv_files, (list, tuple)):
            df = pd.concat([pd.read_csv(f, sep=';', index_col=False) for f in csv_files], ignore_index=True)
        else:'''
        csv_files = Path(MODEL_TRAIN_DATA / 'train_therapeutic_outcome.csv')
        df = pd.read_csv(str(csv_files), sep=';', index_col=False)
        df.columns = [col.lower() for col in df.columns]
        df = df[[c.lower() for c in cols]]
        targets = [t.lower() for t in targets] # Asegurarse de que targets está en minúscula

        # 1. Normaliza todas las columnas target a string (para stratify y LabelEncoder)
        for t in targets:
            df[t] = df[t].astype(str)
        
        # 2. Crear 'stratify_col' A PARTIR DE LOS STRINGS
        df['stratify_col'] = df[targets].astype(str).agg("_".join, axis=1)
        df = df.dropna(subset=targets, axis=0, ignore_index=True)
        
        # 3. Filtrar datos insuficientes (como antes)
        counts = df['stratify_col'].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df['stratify_col'].isin(suficientes)]

        # 4. AHORA, procesar las columnas multi-etiqueta
        def split_labels(label_str):
            """Divide un string como 'A, B' en una lista ['A', 'B'] y maneja nulos."""
            if not isinstance(label_str, str) or label_str.lower() in ['nan', '', 'null']:
                return [] # MultiLabelBinarizer necesita una lista (iterable)
            return [s.strip() for s in label_str.split(',')]

        for col in self.multi_label_cols:
            if col in df.columns:
                df[col] = df[col].apply(split_labels)
        
        # 5. Ajustar los encoders (LabelEncoder y MultiLabelBinarizer)
        self.fit_encoders(df)
        
        # 6. Transformar los datos
        df = self.transform(df)
        
        self.data = df.drop(columns=['stratify_col']).reset_index(drop=True)
        
        # Esta parte parece redundante en tu original, la he eliminado.
        # df = df.dropna(subset=targets, axis=0, ignore_index=True)
        # self.fit_encoders(df, targets)
        # df = self.transform(df, targets)
        # self.data = df.reset_index(drop=True)
        
        return self.data
    
    def get_tensors(self, cols):
        # Esta función parece no usarse en el pipeline de PGenDataset,
        # pero si se usa, necesitaría lógica similar a PGenDataset.__init__
        tensors = {}
        for col in cols:
            if col in self.data:
                if col in self.multi_label_cols:
                    stacked_data = np.stack(self.data[col].values)
                    tensors[col.lower()] = torch.tensor(stacked_data, dtype=torch.float32)
                else:
                    tensors[col.lower()] = torch.tensor(self.data[col].values, dtype=torch.long)
        
        for target in self.target_cols:
             if target in self.data:
                if target in self.multi_label_cols:
                    stacked_data = np.stack(self.data[target].values)
                    tensors[target] = torch.tensor(stacked_data, dtype=torch.float32)
                else:
                    tensors[target] = torch.tensor(self.data[target].values, dtype=torch.long)
        return tensors

class PGenDataset(Dataset):
    def __init__(self, df, target_cols, multi_label_cols: set): 
        """
        Crea tensores para el DataLoader.
        'multi_label_cols' es un set de nombres de columnas que deben ser
        convertidas a tensores Float32 (para BCEWithLogitsLoss).
        """
        self.tensors = {}
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = multi_label_cols or set()

        for col in df.columns:
            if col == 'stratify_col' or col not in df:
                continue
            
            if col in self.multi_label_cols:
                # Para columnas multi-etiqueta (que contienen arrays)
                # Apilarlas en un solo tensor y convertir a Float
                try:
                    stacked_data = np.stack(df[col].values)
                    self.tensors[col] = torch.tensor(stacked_data, dtype=torch.float32)
                except ValueError as e:
                    print(f"Error al apilar la columna multi-etiqueta: {col}")
                    # Esto puede pasar si 'transform' no se ejecutó correctamente
                    # y la columna contiene listas de diferentes longitudes.
                    raise e
            else:
                # Para inputs y targets de etiqueta única
                # Convertir a Long como antes
                try:
                    self.tensors[col] = torch.tensor(df[col].values, dtype=torch.long)
                except TypeError as e:
                    print(f"Error al crear tensor para la columna: {col}. ¿Contiene tipos mixtos?")
                    raise e

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, idx):
        # Devuelve un diccionario de tensores para este índice
        return {k: v[idx] for k, v in self.tensors.items()}

def train_data_load(targets):
    csv_path = MODEL_TRAIN_DATA
    targets = [t.lower() for t in targets]
    # 'read_cols' debe incluir todas las columnas de input Y target
    read_cols_set = set(targets) | {"Drug", "Gene", "Allele", "Genotype"}
    read_cols = [c.lower() for c in read_cols_set]
    
    csvfiles = glob.glob("*.csv") #glob.glob(f"{csv_path}/*.csv")
    equivalencias = load_equivalencias(csv_path)
    return csvfiles, read_cols, equivalencias