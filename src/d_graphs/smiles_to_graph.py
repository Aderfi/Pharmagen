import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import rdchem

# --- FUNCIONES AUXILIARES ---
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

# --- PRUEBA CON PARACETAMOL ---
smiles = "CC(=O)Nc1ccc(O)cc1" # Paracetamol
grafo = smiles_to_graph_complete(smiles)

print("--- REPORTE DEL GRAFO ---")
print(f"Molécula: {smiles}")
print(f"Forma de 'x' (Nodos): {grafo.x.shape}") 
print(f"Forma de 'edge_index' (Conexiones): {grafo.edge_index.shape}")
print(f"Forma de 'edge_attr' (Info Enlaces): {grafo.edge_attr.shape}")

# Vamos a buscar un enlace DOBLE (El C=O del grupo acetilo)
# Sabemos que es Doble (índice 1 en one-hot), No Conjugado, No Anillo.
print("\nEjemplo de vector de enlace (C=O debería tener un 1 en la posición 1):")
# Imprimimos los atributos del segundo enlace (probablemente el C=O)
print(grafo.edge_attr[2])