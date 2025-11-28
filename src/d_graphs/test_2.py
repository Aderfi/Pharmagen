import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

# Función auxiliar: One-Hot Encoding
# Convierte una categoría en una lista de 0s y un 1
def one_hot_encoding(value, possible_values):
    encoding = [0] * len(possible_values)
    if value in possible_values:
        encoding[possible_values.index(value)] = 1
    return encoding

def smiles_to_graph_pro(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # --- 1. CARACTERÍSTICAS DE LOS ÁTOMOS (Matriz X) ---
    atom_features = []
    
    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom.GetAtomicNum()
        # A. Qué elemento es (Simplificado a los más comunes en farma)
        # Esto crea un vector tipo: [1,0,0,0,0...] si es C
        features = one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'F', 'S', 'Cl', 'P', 'Br', 'I'])
        
        # B. Grado del átomo (a cuántos vecinos reales está unido sin contar H implícitos)
        features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
        
        # C. Hibridación (Vital para geometría: plana vs tetraédrica)
        features += one_hot_encoding(atom.GetHybridization(), [
            rdchem.HybridizationType.SP, 
            rdchem.HybridizationType.SP2, 
            rdchem.HybridizationType.SP3
        ])
        
        # D. ¿Es aromático? (0 o 1)
        features.append(1 if atom.GetIsAromatic() else 0)
        
        # E. Masa atómica (Normalizada para que no sea un número gigante comparado con los 0s y 1s)
        features.append(atom.GetMass() * 0.01)

        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # --- 2. CONECTIVIDAD (Edge Index) ---
    edge_indices = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Enlaces bidireccionales
        edge_indices.append([start, end])
        edge_indices.append([end, start])
    
    # Si la molécula es un ion simple (sin enlaces), edge_index debe estar vacío pero con forma correcta
    if len(edge_indices) == 0:
         edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# --- PRUEBA DE CONCEPTO ---
smiles_paracetamol = "CC(=O)Nc1ccc(O)cc1" # Paracetamol
grafo = smiles_to_graph_pro(smiles_paracetamol)

print(f"Fármaco procesado: Paracetamol")
print(f"Número de átomos (nodos): {grafo.num_nodes}") 
print(f"Número de características por átomo: {grafo.num_node_features}")
print(f"Número de 'caminos' (enlaces x2): {grafo.num_edges}")

# Inspección de un átomo (El primero, un Carbono del metilo)
print(f"\nVector del primer átomo:\n{grafo.x[0]}")