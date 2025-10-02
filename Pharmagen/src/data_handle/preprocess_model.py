"""
Funciones y clases para el preprocesamiento y tokenización de datos
para el modelo predictivo de Pharmagen.
Convierte listas de mutaciones y medicamentos en secuencias de enteros
(los embeddings del modelo) a partir de vocabularios únicos.
Autor: Astordna / Aderfi / Adrim Hamed Outmani
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def build_vocab(sequences: List[str], sep: str = ",") -> Dict[str, int]:
    """
    Construye un vocabulario (token->índice) a partir de una lista de secuencias separadas por sep.
    El índice 0 se reserva para el padding.
    """
    vocab = set()
    for seq in sequences:
        tokens = seq.split(sep)
        vocab.update(token.strip() for token in tokens if token.strip())
    return {token: idx + 1 for idx, token in enumerate(sorted(vocab))}  # 0 es padding

def texts_to_sequences(series: pd.Series, vocab: Dict[str, int], sep: str = ",") -> List[List[int]]:
    """
    Convierte una columna de texto (listas separadas por sep) en listas de enteros usando el vocabulario.
    """
    result = []
    for row in series:
        tokens = row.split(sep)
        indices = [vocab.get(token.strip(), 0) for token in tokens]  # 0 para OOV/padding
        result.append(indices)
    return result

def pad_sequences_list(sequences: List[List[int]], maxlen: Optional[int] = None) -> np.ndarray:
    """
    Aplica padding post a una lista de secuencias de enteros.
    """
    return pad_sequences(sequences, maxlen=maxlen, padding='post', value=0)

class PharmagenPreprocessor:
    """
    Clase para preprocesar datos de mutaciones y medicamentos:
    - Construye vocabularios
    - Convierte a secuencias de enteros
    - Aplica padding
    """
    def __init__(self, mut_sep: str = ",", drug_sep: str = ","):
        self.mut_sep = mut_sep
        self.drug_sep = drug_sep
        self.mut_vocab: Dict[str, int] = {}
        self.drug_vocab: Dict[str, int] = {}
        self.maxlen_mut: Optional[int] = None
        self.maxlen_drug: Optional[int] = None

    def fit(self, df: pd.DataFrame, mut_col: str, drug_col: str):
        """
        Construye los vocabularios y determina la longitud máxima de cada input.
        """
        self.mut_vocab = build_vocab(df[mut_col].astype(str).tolist(), self.mut_sep)
        self.drug_vocab = build_vocab(df[drug_col].astype(str).tolist(), self.drug_sep)
        self.maxlen_mut = df[mut_col].astype(str).apply(lambda x: len(x.split(self.mut_sep))).max()
        self.maxlen_drug = df[drug_col].astype(str).apply(lambda x: len(x.split(self.drug_sep))).max()
    
    def transform(self, df: pd.DataFrame, mut_col: str, drug_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tokeniza y realiza padding de los datos.
        """
        X_mut = texts_to_sequences(df[mut_col].astype(str), self.mut_vocab, self.mut_sep)
        X_drug = texts_to_sequences(df[drug_col].astype(str), self.drug_vocab, self.drug_sep)
        X_mut_pad = pad_sequences_list(X_mut, self.maxlen_mut)
        X_drug_pad = pad_sequences_list(X_drug, self.maxlen_drug)
        return X_mut_pad, X_drug_pad

    def fit_transform(self, df: pd.DataFrame, mut_col: str, drug_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ajusta los vocabularios y transforma los datos en una sola llamada.
        """
        self.fit(df, mut_col, drug_col)
        return self.transform(df, mut_col, drug_col)

    def save_vocabs(self, mut_path: str, drug_path: str):
        """
        Guarda los diccionarios de vocabulario como archivos .json.
        """
        import json
        with open(mut_path, 'w', encoding='utf-8') as f:
            json.dump(self.mut_vocab, f, ensure_ascii=False, indent=4)
        with open(drug_path, 'w', encoding='utf-8') as f:
            json.dump(self.drug_vocab, f, ensure_ascii=False, indent=4)

    def load_vocabs(self, mut_path: str, drug_path: str):
        """
        Carga los diccionarios de vocabulario desde archivos .json.
        """
        import json
        with open(mut_path, 'r', encoding='utf-8') as f:
            self.mut_vocab = json.load(f)
        with open(drug_path, 'r', encoding='utf-8') as f:
            self.drug_vocab = json.load(f)