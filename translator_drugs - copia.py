import torch
import lmdb
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
import json
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem

# Nota: shelve no soporta compresión gzip nativa directa sobre el archivo DB, 
# pero es lo más eficiente para acceso aleatorio.
# Si el espacio es crítico, el enfoque cambia (ver abajo).
def one_hot_encoding(value, possible_values):
    encoding = [0] * len(possible_values)
    if value in possible_values:
        encoding[possible_values.index(value)] = 1
    return encoding

def smiles_to_graph_complete(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # -----------------------------------------------------
    # 1. NODOS (Átomos) - Igual que antes
    # -----------------------------------------------------
    atom_features = []
    for atom in mol.GetAtoms():
        features = one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Br', 'I'])
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        features += one_hot_encoding(atom.GetHybridization(), [
            rdchem.HybridizationType.SP, rdchem.HybridizationType.SP2, rdchem.HybridizationType.SP3
        ])
        features.append(1 if atom.GetIsAromatic() else 0)
        features.append(atom.GetMass() * 0.01)
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # -----------------------------------------------------
    # 2. ARISTAS (Enlaces) - Aquí está la novedad
    # -----------------------------------------------------
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # A. Extraer Tipo de Enlace (One-Hot)
        # Vector de 4 posiciones: [Simple, Doble, Triple, Aromático]
        bt = bond.GetBondType()
        bond_feats = one_hot_encoding(bt, [
            rdchem.BondType.SINGLE, 
            rdchem.BondType.DOUBLE, 
            rdchem.BondType.TRIPLE, 
            rdchem.BondType.AROMATIC
        ])
        
        # B. ¿Está conjugado? (Importante para resonancia/color/estabilidad)
        bond_feats.append(1 if bond.GetIsConjugated() else 0)
        
        # C. ¿Está en un anillo?
        bond_feats.append(1 if bond.IsInRing() else 0)

        # --- SINCRONIZACIÓN CRÍTICA ---
        # Como el grafo es no dirigido, añadimos la ida y la vuelta.
        # Ambas direcciones comparten las mismas características químicas.
        
        # Dirección 1: Start -> End
        edge_indices.append([start, end])
        edge_attrs.append(bond_feats)
        
        # Dirección 2: End -> Start
        edge_indices.append([end, start])
        edge_attrs.append(bond_feats)

    # Convertir a Tensores
    if len(edge_indices) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float) # 6 características por enlace
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


df_dict = json.load(open('cid_smiles_dict.json', 'r'))

def create_lmdb_library(json_path, lmdb_path, map_size=1099511627776): # 1 TB
    """
    Crea una base de datos LMDB a partir de un diccionario JSON de SMILES.
    map_size: Tamaño máximo virtual de la DB (1TB por defecto para seguridad).
    """
    
    # 1. Cargar diccionario de SMILES
    print("Cargando diccionario JSON...")
    with open(json_path, 'r') as f:
        df_dict = json.load(f)

    # 2. Abrir entorno LMDB
    # map_size debe ser mayor que lo que esperas guardar. 
    # No usa espacio real en disco hasta que escribes datos.
    env = lmdb.open(lmdb_path, map_size=map_size)

    print(f"Generando grafos y guardando en {lmdb_path}...")
    
    # Usamos write=True para escribir
    with env.begin(write=True) as txn:
        for cid, smiles in tqdm(df_dict.items(), total=len(df_dict)):
            
            # A. Conversión Química -> Matemática
            graph_data = smiles_to_graph_complete(smiles)
            
            if graph_data is not None:
                # B. Preparar el objeto para guardar
                # Es mejor guardar el objeto Data() completo que un diccionario suelto,
                # así PyG lo entiende nativamente al cargarlo.
                
                # C. Serialización (Objeto -> Bytes)
                # Usamos pickle.dumps. Protocolo -1 usa la versión más alta disponible.
                value_bytes = pickle.dumps(graph_data, protocol=-1)
                
                # D. Clave (Debe ser bytes también)
                key_bytes = str(cid).encode('ascii')
                
                # E. Guardar en la transacción
                txn.put(key_bytes, value_bytes)

    env.close()
    print("Biblioteca LMDB creada exitosamente.")

if __name__ == "__main__":
    create_lmdb_library('cid_smiles_dict.json', 'drug_graphs/drug_library.lmdb')
    print("Proceso terminado. Los datos están en 'drug_graphs/drug_library.lmdb'.")

