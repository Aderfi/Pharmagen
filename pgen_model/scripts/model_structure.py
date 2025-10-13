import torch
import torch.nn as nn
import torch.optim as optim
from . import model_structure

class PharmacoModel(nn.Module):
    def __init__(self, n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=8, hidden_dim=64):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.effect_out = nn.Linear(hidden_dim//2, n_effects)
        self.outcome_out = nn.Linear(hidden_dim//2, n_outcomes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, drug, genotype):
        drug_e = self.drug_emb(drug)
        geno_e = self.geno_emb(genotype)
        x = torch.cat([drug_e, geno_e], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        effect = self.effect_out(x)
        outcome = self.outcome_out(x)
        return effect, outcome