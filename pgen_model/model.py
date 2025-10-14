import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PGenModel(nn.Module):
    def __init__(self, n_drug_genos, emb_dim, hidden_dim, dropout_rate, output_dims):
        """
        n_drug_genos: número de combinaciones únicas de Drug_Geno
        output_dims: dict con nombre de columna como clave y número de clases como valor,
        ej: { "outcome": 4, "variation": 3 }
        """
        super().__init__()
        self.druggeno_emb = nn.Embedding(n_drug_genos, emb_dim).to(device)
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        
        # Crear solo las salidas necesarias
        self.output_layers = nn.ModuleDict({
            name: nn.Linear(hidden_dim // 2, n_classes)
            for name, n_classes in output_dims.items()
        })
        self.output_names = list(output_dims.keys())

    def forward(self, drug_geno):
        x = self.druggeno_emb(drug_geno)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        output = {name: self.output_layers[name](x) for name in self.output_names}
        return output
