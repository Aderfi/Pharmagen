import glob
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import src.config.config as cfg
import torch
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from src.config.config import *
from torch.utils.data import Dataset
from .model_configs import MULTI_LABEL_COLUMN_NAMES


def train_data_import(targets):
    csv_path = MODEL_TRAIN_DATA
    
    
    csv_files = glob.glob(f"{csv_path}/*.tsv")

    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos TSV en: {csv_path}")
    elif len(csv_files) == 1:
        print(f"Se encontró un único archivo TSV. Usando: {csv_files[0]}")
        csv_files = csv_files[0]
    csv_files = Path(csv_path, "final_test_genalle.tsv")
            
    targets = [t.lower() for t in targets]

    read_cols_set = set(targets) | {"Drug", "Genalle", "Gene", "Allele"} #Variant/Haplotypes cambiado a Genalle
    read_cols = [c.lower() for c in read_cols_set]

    # equivalencias = load_equivalencias(csv_path)
    return csv_files, read_cols

def load_equivalencias(csv_path):
    with open(f"{csv_path}/json_dicts/geno_alleles_dict.json", "r") as f:
        equivalencias = json.load(f)

        for k, v in equivalencias.items():
            if re.match(r"rs[0-9]+", k):
                continue
            else:
                equivalencias[k] = np.nan

        equivalencias = (
            equivalencias.pop(key) if (key := "nan") in equivalencias else equivalencias
        )

    return equivalencias

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


class PGenDataProcess:
    """
    Clase refactorizada para preprocesar datos de farmacogenética.
    
    Esta clase ahora sigue el flujo de trabajo fit/transform de Scikit-learn
    para prevenir la fuga de datos (data leakage).
    
    1. Llama a `load_and_clean_data` para cargar y limpiar el CSV.
    2. Divide los datos (train/val/test) FUERA de esta clase.
    3. Llama a `fit` SÓLO con los datos de entrenamiento (train_df).
    4. Llama a `transform` para aplicar el preprocesamiento a 
       train_df, val_df, y test_df.
    """

    def __init__(self):
        self.encoders = {}
        self.cols_to_process = []
        self.target_cols = []
        self.multi_label_cols = set()

        self.unknown_token = "__UNKNOWN__" 

    def _split_labels(self, label_str):
        """Helper: Divide un string 'A, B' en una lista ['A', 'B'] y maneja nulos."""
        if not isinstance(label_str, str) or pd.isna(label_str) or label_str.lower() in ["nan", "", "null"]:
            return []  # MultiLabelBinarizer necesita un iterable
        return [s.strip() for s in re.split(r",|;", label_str) if s.strip()]

    def load_data(self, csv_path, all_cols, target_cols, 
                            multi_label_targets, stratify_cols):
        """
        Carga, limpia y prepara los datos para la división (splitting).
        NO ajusta ningún encoder.
        """
        self.cols_to_process = [c.lower() for c in all_cols]
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = set(t.lower() for t in multi_label_targets or [])
        
        df = pd.read_csv(str(csv_path), sep="\t", index_col=False)
        df.columns = [col.lower() for col in df.columns]
        
        df = df[self.cols_to_process]
        
        print("Limpiando nombres de entidades (reemplazando espacios con '_')...")
        # Vectorized string replacement is much faster than apply()
        for col in self.cols_to_process:
            if col in df.columns and col not in self.multi_label_cols:
                # Reemplazar espacios en columnas single-label (vectorized)
                df[col] = df[col].astype(str).str.replace(' ', '_', regex=False)
            elif col in df.columns and col in self.multi_label_cols:
                # Reemplazar espacios en columnas multi-label
                # (Asumiendo que _split_labels ya las convirtió a listas de strings)
                 df[col] = df[col].apply(
                     lambda lst: [s.replace(' ', '_') for s in lst]
                 )
        
        task3_name = "effect_type"
        if task3_name in df.columns:
            print(f"Agrupando clases raras para '{task3_name}'...")
            MIN_SAMPLES = 20 # Umbral: agrupar cualquier clase con < 20 muestras
            
            # 1. Obtener los conteos de clase
            counts = df[task3_name].value_counts()
            
            # 2. Identificar las clases a agrupar (using set for faster lookup)
            to_group = set(counts[counts < MIN_SAMPLES].index)
            
            if len(to_group) > 0:
                print(f"Se agruparán {len(to_group)} clases en 'Other_Grouped'.")
                # 3. Reemplazarlas en el DataFrame (vectorized with map)
                df[task3_name] = df[task3_name].map(
                    lambda x: "Other_Grouped" if x in to_group else x
                )
            else:
                print("No se encontraron clases raras para agrupar.")
                
        # 1. Normaliza columnas target (y de estratificación) a string
        for t in self.target_cols:
            df[t] = df[t].astype(str)
            
        for s_col in stratify_cols:
             df[s_col] = df[s_col].astype(str)

        # 2. Crear 'stratify_col'
        # Usamos las columnas de estratificación proporcionadas
        df["stratify_col"] = df[stratify_cols].agg("_".join, axis=1)

        # 3. Filtrar datos insuficientes (opcional, pero buena práctica)
        # Use boolean indexing instead of isin for better performance
        counts = df["stratify_col"].value_counts()
        suficientes = counts[counts > 1].index
        df = df[df["stratify_col"].isin(suficientes)].reset_index(drop=True)

        # 4. Procesar las columnas multi-etiqueta (convertir string a lista)
        for col in self.multi_label_cols:
            if col in df.columns:
                df[col] = df[col].apply(self._split_labels)
        
        print(f"Datos cargados y limpios. Total de filas: {len(df)}")
        return df

    def fit(self, df_train):
        """
        <--- NUEVO: Ajusta los encoders SÓLO con los datos de entrenamiento.
        """
        print("Ajustando encoders con datos de entrenamiento...")
        for col in self.cols_to_process:
            if col in self.multi_label_cols:
                # MultiLabelBinarizer: se ajusta a las listas de etiquetas
                encoder = MultiLabelBinarizer()
                encoder.fit(df_train[col])
                self.encoders[col] = encoder
            else:
                # LabelEncoder: se ajusta a etiquetas únicas
                encoder = LabelEncoder()
                
                # <--- LÓGICA MEJORADA ---
                # Añadir un token "UNKNOWN" al vocabulario del encoder
                # para manejar valores en 'val' o 'test' que no estaban en 'train'.
                all_labels = list(df_train[col].astype(str).unique())
                if self.unknown_token not in all_labels:
                    all_labels.append(self.unknown_token)
                
                encoder.fit(all_labels)
                self.encoders[col] = encoder
        print("Encoders ajustados.")

    def transform(self, df):
        """
        <--- NUEVO: Transforma un DataFrame (train, val, o test)
        usando los encoders ya ajustados.
        """
        df_transformed = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df_transformed.columns:
                continue

            if isinstance(encoder, MultiLabelBinarizer):
                # MLB transforma las listas en vectores binarios
                # (Ignora etiquetas que no vio en 'fit', lo cual es correcto)
                df_transformed[col] = list(encoder.transform(df_transformed[col]))
            
            elif isinstance(encoder, LabelEncoder):
                # <--- LÓGICA MEJORADA ---
                # 1. Obtener las etiquetas que el encoder conoce (de 'fit')
                known_labels = set(encoder.classes_)
                
                # 2. Reemplazar cualquier etiqueta no conocida por el token UNKNOWN
                original_col = df_transformed[col].astype(str)
                transformed_col = original_col.apply(
                    lambda x: x if x in known_labels else self.unknown_token
                )
                
                # 3. Ahora, transformar de forma segura
                df_transformed[col] = encoder.transform(transformed_col)

        return df_transformed



class PGenDataset(Dataset):
    def __init__(self, df, target_cols, multi_label_cols=None):
        """
        Crea tensores para el DataLoader.
        'multi_label_cols' es un set de nombres de columnas que deben ser
        convertidas a tensores Float32 (para BCEWithLogitsLoss).
        """
        self.tensors = {}
        self.target_cols = [t.lower() for t in target_cols]
        self.multi_label_cols = multi_label_cols or set()

        for col in df.columns:
            if col == "stratify_col" or col not in df:
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
                    print(
                        f"Error al crear tensor para la columna: {col}. ¿Contiene tipos mixtos?"
                    )
                    raise e

    def __len__(self):
        return len(next(iter(self.tensors.values())))

    def __getitem__(self, idx):
        # Devuelve un diccionario de tensores para este índice
        return {k: v[idx] for k, v in self.tensors.items()}


