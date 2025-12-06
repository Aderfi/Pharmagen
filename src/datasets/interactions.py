"""
interactions.py

Handles the pairing of Drugs and Genomic entities for training.
Provides the Dataset and Collation logic for the Graph-based model.
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from typing import List, Dict, Any, Tuple

class GraphInteractionDataset(Dataset):
    """
    Dataset for Drug-Gene interactions.
    
    Args:
        interactions_df: DataFrame containing 'drug_id', 'chrom', 'pos', and target columns.
        drug_dataset: Pre-computed/loaded DrugGraphDataset (indexable by drug_id or index).
        gene_dataset: Pre-computed/loaded GeneEmbeddingDataset.
        target_cols: List of target column names.
    """
    def __init__(self, interactions_df, drug_dataset, gene_dataset, target_cols):
        self.df = interactions_df.reset_index(drop=True)
        self.drug_dataset = drug_dataset
        self.gene_dataset = gene_dataset
        self.target_cols = target_cols
        
        # Optimization: Map Drug IDs to indices in drug_dataset if necessary
        # Assuming drug_dataset is indexable by the ID present in df for O(1) lookup
        # Or we create a mapping here.
        self.drug_id_to_idx = {id_: i for i, id_ in enumerate(drug_dataset.drug_identifiers)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Get Drug Graph
        drug_id = str(row['drug']) # Assuming column name 'drug'
        if drug_id in self.drug_id_to_idx:
            drug_idx = self.drug_id_to_idx[drug_id]
            drug_graph = self.drug_dataset[drug_idx]
        else:
            # Fallback or Error. For now, returning None (collator must handle)
            drug_graph = None

        # 2. Get Gene Embedding
        # Gene dataset expects dict with 'chrom', 'pos'
        gene_query = {'chrom': str(row['chromosome']), 'pos': int(row['position'])}
        # Gene dataset might calculate on fly or look up. 
        # If GeneDataset is list-based, we need to map row to index.
        # Assuming GeneDataset can handle the query directly or we pass the index.
        # Let's assume GeneDataset is wrapper around the whole genome and takes coords.
        # Actually, gene_dataset.__getitem__ takes an index. 
        # We should probably pass the specific query to a method, or assume 1-to-1 mapping isn't pre-defined.
        # Let's Refactor GeneEmbeddingDataset slightly to allow ad-hoc queries or just call the method.
        
        # Direct call to compute/fetch embedding (KISS)
        gene_emb = self.gene_dataset.get_embedding_for_coords(
            gene_query['chrom'], gene_query['pos']
        )

        # 3. Targets
        targets = {col: torch.tensor(row[col], dtype=torch.float) for col in self.target_cols}
        
        return drug_graph, gene_emb, targets

    
def graph_interaction_collator(batch: List[Tuple]):
    """
    Custom collator to batch Graphs (PyG) and Stack Tensors.
    """
    drug_graphs = []
    gene_embs = []
    targets_list = []
    
    for d_graph, g_emb, t in batch:
        if d_graph is None: continue # Skip invalid drugs
        drug_graphs.append(d_graph)
        gene_embs.append(g_emb)
        targets_list.append(t)
        
    if not drug_graphs:
        return None
        
    # Batch Graphs
    drug_batch = Batch.from_data_list(drug_graphs)
    
    # Stack Embeddings
    gene_batch = torch.stack(gene_embs)
    
    # Stack Targets
    targets_batch = {}
    if targets_list:
        keys = targets_list[0].keys()
        for k in keys:
            targets_batch[k] = torch.stack([item[k] for item in targets_list])
            
    return {
        "drug_data": drug_batch,
        "gene_input": gene_batch,
        "targets": targets_batch
    }

