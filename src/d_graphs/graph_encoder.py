import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        # GCN simple pero efectiva para proyectar química a espacio latente
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim) # Output debe ser == embedding_dim
        self.act = nn.GELU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.act(self.conv1(x, edge_index))
        x = self.act(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index) # Sin activación final para permitir rango completo
        
        # Pooling: [Total_Atomos, Dim] -> [Batch_Size, Dim]
        return global_mean_pool(x, batch)