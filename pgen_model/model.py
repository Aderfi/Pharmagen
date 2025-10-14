import torch
import torch.nn as nn

class PGenModel(nn.Module):
    def __init__(self, n_drugs, n_genotypes, emb_dim, hidden_dim, dropout_rate, output_dims):
        """
        output_dims: dict con nombre de columna como clave y número de clases como valor,
        ej: { "outcome": 4, "variation": 3 }
        """
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Crear solo las salidas necesarias
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(hidden_dim // 2, n_classes)
            for name, n_classes in output_dims.items()
        })
        self.output_names = list(output_dims.keys())

    def forward(self, drug, genotype):
        x = torch.cat([self.drug_emb(drug), self.geno_emb(genotype)], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = {name: self.output_layers[name](x) for name in self.output_names}
        return output