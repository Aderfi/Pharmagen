"""
modeling_graph.py

Alternative modeling structure based on PyTorch Geometric.
Handles Graph inputs for Drugs and Embeddings/Graphs for Genomics.

Adheres to: Zen of Python, SOLID, KISS.
"""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Batch

# =============================================================================
# COMPONENT: GNN ENCODER (For Drugs)
# =============================================================================

class DrugGNNEncoder(nn.Module):
    """
    Graph Neural Network encoder for molecular graphs.
    Uses GAT (Graph Attention Network) or GCN layers.
    """
    def __init__(self, in_channels: int, hidden_dim: int, out_dim: int, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input Layer
        self.layers.append(GATConv(in_channels, hidden_dim))
        
        # Hidden Layers
        for _ in range(n_layers - 1):
            self.layers.append(GATConv(hidden_dim, hidden_dim))
            
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        # Message Passing
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global Pooling (Readout)
        # Combine Mean and Max pooling for better representation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = x_mean + x_max
        
        # Projection
        x = self.out_proj(x)
        return x

# =============================================================================
# COMPONENT: GENOME HEAD (Embedding Processor)
# =============================================================================

class GenomeEncoder(nn.Module):
    """
    Processes genomic embeddings (e.g., from Nucleotide Transformer).
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim)
        )

    def forward(self, x):
        return self.net(x)

# =============================================================================
# MAIN MODEL: GRAPH PHARMAGEN
# =============================================================================

class PGenGraphModel(nn.Module):
    """
    Multi-modal Graph Neural Network for Pharmacogenomics.
    
    Inputs:
        - drug_graph: PyG Batch object (Nodes, Edges, Attributes)
        - genome_input: Tensor (Embeddings)
        
    Output:
        - Predictions for multiple target heads.
    """
    def __init__(
        self,
        drug_node_features: int,
        gene_input_dim: int,
        target_dims: Dict[str, int], # Output heads (e.g., {'phenotype': 1})
        hidden_dim: int = 256,
        embedding_dim: int = 128,
        n_gnn_layers: int = 3,
        dropout_rate: float = 0.2
    ):
        super().__init__()
        
        # 1. Encoders
        self.drug_encoder = DrugGNNEncoder(
            in_channels=drug_node_features,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            n_layers=n_gnn_layers,
            dropout=dropout_rate
        )
        
        self.gene_encoder = GenomeEncoder(
            in_dim=gene_input_dim,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            dropout=dropout_rate
        )
        
        # 2. Fusion
        fusion_dim = embedding_dim * 2
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 3. Output Heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim) 
            for name, dim in target_dims.items()
        })
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, drug_data: Batch, gene_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            drug_data: Batch object from torch_geometric
            gene_input: Tensor [Batch, Gene_Dim]
        """
        # Encode Modalities
        drug_vec = self.drug_encoder(drug_data.x, drug_data.edge_index, drug_data.batch) # type: ignore
        gene_vec = self.gene_encoder(gene_input)
        
        # Fusion
        combined = torch.cat([drug_vec, gene_vec], dim=1)
        fused = self.fusion_layer(combined)
        
        # Outputs
        return {name: head(fused) for name, head in self.heads.items()}

