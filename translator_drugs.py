import pubchempy as pcp
import pandas as pd
import os
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem
from tqdm import tqdm

# ==========================================
# CONFIGURACIÓN (EDITAR AQUÍ)
# ==========================================
INPUT_CSV = "farmacos_extraidos.tsv"  # Nombre de tu archivo CSV
OUTPUT_FILE = "biblioteca_grafos.pt" # Nombre del archivo de salida (.pt es estándar en PyTorch)

# Nombre de la columna en el CSV que contiene los SMILES
# (Abre tu CSV y verifica si se llama "smiles", "SMILES", "structure", etc.)
SMILES_COL = "smiles" 

# Opcional: Nombre de la columna que contiene la etiqueta a predecir (actividad, toxicidad, etc.)
LABEL_COL = None  # Cambia a None si no tienes etiqueta

# ==========================================
# TUS FUNCIONES DE QUÍMICA
# ==========================================

def one_hot_encoding(value, possible_values):
    encoding = [0] * len(possible_values)
    if value in possible_values:
        encoding[possible_values.index(value)] = 1
    return encoding

def smiles_to_graph_complete(smiles, label=None):
    """
    Convierte SMILES a objeto PyG Data con features avanzadas.
    Acepta una etiqueta opcional (y) para aprendizaje supervisado.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: 
        return None

    # --- 1. NODOS (Átomos) ---
    atom_features = []
    for atom in mol.GetAtoms():
        # Símbolo
        features = one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Br', 'I'])
        # Grado (nº vecinos)
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        # Hibridación
        features += one_hot_encoding(atom.GetHybridization(), [
            rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3
        ])
        # Aromaticidad
        features.append(1 if atom.GetIsAromatic() else 0)
        # Masa normalizada (aprox)
        features.append(atom.GetMass() * 0.01)
        
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # --- 2. ARISTAS (Enlaces) ---
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Features del enlace
        bt = bond.GetBondType()
        bond_feats = one_hot_encoding(bt, [
            rdchem.BondType.SINGLE, 
            rdchem.BondType.DOUBLE, 
            rdchem.BondType.TRIPLE, 
            rdchem.BondType.AROMATIC
        ])
        bond_feats.append(1 if bond.GetIsConjugated() else 0)
        bond_feats.append(1 if bond.IsInRing() else 0)

        # Grafo no dirigido: Añadimos ida (start->end) y vuelta (end->start)
        edge_indices.append([start, end])
        edge_attrs.append(bond_feats)
        
        edge_indices.append([end, start])
        edge_attrs.append(bond_feats)

    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    # Creamos el objeto Data. 
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    
    # Si tenemos una etiqueta (ej. toxicidad: 0 o 1), la añadimos al grafo
    if label is not None:
        # Asumimos que es un problema de clasificación o regresión (float)
        data.y = torch.tensor([label], dtype=torch.float)
        
    return data

# ==========================================
# LÓGICA DE PROCESAMIENTO CSV
# ==========================================

def process_csv_to_graphs():
    print(f"Leyendo archivo: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV, sep='\t')
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return

    # Verificar columnas
    if SMILES_COL not in df.columns:
        print(f"❌ Error: La columna '{SMILES_COL}' no existe en el CSV.")
        print(f"Columnas encontradas: {list(df.columns)}")
        return

    graph_list = []
    failed_mols = 0

    print("Iniciando conversión de moléculas a grafos...")
    
    # 3. Iterar y Convertir
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        smiles = row[SMILES_COL]
        
        # Obtener etiqueta si existe configuración
        label = row[LABEL_COL] if LABEL_COL and LABEL_COL in df.columns else None
        
        try:
            graph = smiles_to_graph_complete(smiles, label)
            
            if graph is not None:
                graph_list.append(graph)
            else:
                failed_mols += 1
        except Exception:
            failed_mols += 1

    # 4. Guardar resultado
    print(f"\nGuardando {len(graph_list)} grafos en {OUTPUT_FILE}...")
    torch.save(graph_list, OUTPUT_FILE)
    
    print("-" * 30)
    print("PROCESO TERMINADO")
    print(f"✅ Grafos creados correctamente: {len(graph_list)}")
    print(f"❌ Fallos (SMILES inválidos): {failed_mols}")
    print(f"📁 Archivo guardado: {os.path.abspath(OUTPUT_FILE)}")
    
    # Ejemplo de cómo cargar
    print("\n--- Cómo cargar estos datos después ---")
    print(f"mis_grafos = torch.load('{OUTPUT_FILE}')")
    print(f"print(mis_grafos[0])")

if __name__ == "__main__":
    process_csv_to_graphs()