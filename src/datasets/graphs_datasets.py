"""
graphs_datasets.py

Handles the loading and processing of drug data for graph-based modeling.
Maps drug identifiers (Name, CID, SID) to SMILES using PubChemPy and converts them
to PyTorch Geometric graph data objects.

Adheres to: Zen of Python, SOLID, KISS.
"""

import os
import logging
from typing import List, Optional, Union, Dict, Any
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem import rdchem
import pubchempy as pcp

# Configure Logging
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITIES: SMILES TO GRAPH
# =============================================================================

def one_hot_encoding(value: Any, possible_values: List[Any]) -> List[int]:
    """Performs one-hot encoding for a value given a list of possibilities."""
    encoding = [0] * len(possible_values)
    if value in possible_values:
        encoding[possible_values.index(value)] = 1
    return encoding

def smiles_to_graph_complete(smiles: str) -> Optional[Data]:
    """
    Converts a SMILES string to a PyTorch Geometric Data object (Graph).
    
    Args:
        smiles: The SMILES string representation of the molecule.
        
    Returns:
        torch_geometric.data.Data object or None if conversion fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning(f"Could not create RDKit molecule from SMILES: {smiles}")
        return None

    # -----------------------------------------------------
    # 1. NODES (Atoms)
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
    # 2. EDGES (Bonds)
    # -----------------------------------------------------
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Bond Type
        bt = bond.GetBondType()
        bond_feats = one_hot_encoding(bt, [
            rdchem.BondType.SINGLE, 
            rdchem.BondType.DOUBLE, 
            rdchem.BondType.TRIPLE, 
            rdchem.BondType.AROMATIC
        ])
        
        # Conjugation & Ring Status
        bond_feats.append(1 if bond.GetIsConjugated() else 0)
        bond_feats.append(1 if bond.IsInRing() else 0)

        # Undirected Graph: Add both directions
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

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# =============================================================================
# DATASET HANDLING
# =============================================================================

class DrugGraphDataset(Dataset):
    """
    Dataset that handles mapping of drug identifiers to Graphs.
    Supports caching of processed graphs.
    """
    def __init__(self, root: str, drug_list_file: str, transform=None, pre_transform=None):
        self.drug_list_file = drug_list_file
        self.drug_identifiers = self._load_drug_list(drug_list_file)
        super().__init__(root, transform, pre_transform)
        # Ensure raw_dir exists (managed by super, but we might need custom logic)

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(self.drug_list_file)]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'data_{i}.pt' for i in range(len(self.drug_identifiers))]

    def _load_drug_list(self, file_path: str) -> List[str]:
        """Loads drug identifiers from a text file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Drug list file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def download(self):
        # No downloading of external datasets, we process the input list.
        pass

    def process(self):
        """
        Main processing loop:
        1. Iterate over drug identifiers.
        2. Resolve to SMILES (Name/CID/SID -> SMILES).
        3. Convert SMILES to Graph.
        4. Save Data object.
        """
        import tqdm
        
        data_list = []
        idx = 0
        
        logger.info("Starting drug processing...")
        for drug_id in tqdm.tqdm(self.drug_identifiers, desc="Processing Drugs"):
            
            smiles = self._resolve_to_smiles(drug_id)
            
            if not smiles:
                logger.warning(f"Skipping {drug_id}: No SMILES found.")
                continue
                
            graph_data = smiles_to_graph_complete(smiles)
            
            if graph_data:
                if self.pre_filter is not None and not self.pre_filter(graph_data):
                    continue
                if self.pre_transform is not None:
                    graph_data = self.pre_transform(graph_data)
                
                torch.save(graph_data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
                idx += 1
            else:
                logger.warning(f"Skipping {drug_id}: Graph generation failed.")

    def _resolve_to_smiles(self, identifier: str) -> Optional[str]:
        """
        Resolves a drug identifier to a SMILES string using PubChemPy.
        Handles Name, CID, and SID.
        """
        identifier = str(identifier)
        try:
            compound = None
            
            # Check if it's a CID (numeric)
            if identifier.isdigit():
                # Assume CID first
                try:
                    compound = pcp.Compound.from_cid(int(identifier))
                except:
                    # If not CID, try SID? Usually SID is also numeric but handled differently.
                    # For simplicity, assuming numeric input is CID.
                    pass
            
            # If not numeric or CID failed, try Name
            if compound is None:
                compounds = pcp.get_compounds(identifier, 'name')
                if compounds:
                    compound = compounds[0]
            
            if compound:
                return compound.isomeric_smiles
            else:
                # Try SID lookup if not found yet?
                # pcp.get_substances(identifier, 'sid') ... extraction is harder
                pass
                
            return None

        except Exception as e:
            logger.error(f"Error resolving {identifier}: {e}")
            return None

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))

