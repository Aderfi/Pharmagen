import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from smiles_to_graph import smiles_to_graph_complete

# 1. Tu lista de fármacos (Dataset simulado)
lista_smiles = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirina (13 átomos)
    "CC(=O)Nc1ccc(O)cc1",     # Paracetamol (11 átomos)
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O" # Ibuprofeno (15 átomos)
]

# 2. Convertimos todos a Grafos Data()
lista_grafos = [smiles_to_graph_complete(smi) for smi in lista_smiles]

# Filtramos si alguno falló (dio None)
lista_grafos = [g for g in lista_grafos if g is not None]

# ---------------------------------------------------------
# 3. EL DATALOADER (Aquí ocurre el Diagonal Stacking)
# ---------------------------------------------------------
# batch_size=3 significa que procesaremos las 3 moléculas a la vez
loader = DataLoader(lista_grafos, batch_size=3, shuffle=False)

# 4. Inspeccionar el Lote
for lote in loader:
    print("--- DENTRO DEL LOTE ---")
    print(f"Objeto Batch: {lote}")
    
    # Análisis de lo que ha pasado
    print(f"\n1. Matriz X (Átomos apilados): {lote.x.shape}")
    # Debería ser la suma: 13 + 11 + 15 = 39 átomos totales
    # Shape esperada: [39, 19] (39 átomos, 19 características cada uno)
    
    print(f"2. Vector 'batch' (El índice): {lote.batch}")
    # Verás una secuencia de 0s, luego 1s, luego 2s.
    
    print(f"3. Matriz Edge_Index (Conexiones apiladas): {lote.edge_index.shape}")
    # Contiene todos los enlaces de las 3 moléculas, pero re-indexados
    # para que no se crucen.
    
    # Comprobación de veracidad:
    num_atomos_reales = 13 + 11 + 15
    print(f"\n¿Coincide el tamaño? {'SÍ' if lote.x.shape[0] == num_atomos_reales else 'NO'}")