import torch
import torch.nn as nn

class PGenModel(nn.Module):
    def __init__(self, n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entities, emb_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim)
        self.fc1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.outcome_out = nn.Linear(hidden_dim // 2, n_outcomes) if n_outcomes else None
        self.variation_out = nn.Linear(hidden_dim // 2, n_variations) if n_variations else None
        self.effect_out = nn.Linear(hidden_dim // 2, n_effects) if n_effects else None
        self.entity_out = nn.Linear(hidden_dim // 2, n_entities) if n_entities else None
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, drug, genotype):
        x = torch.cat([self.drug_emb(drug), self.geno_emb(genotype)], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return {
            'outcome': self.outcome_out(x) if self.outcome_out else None,
            'variation': self.variation_out(x) if self.variation_out else None,
            'effect': self.effect_out(x) if self.effect_out else None,
            'entity': self.entity_out(x) if self.entity_out else None
        }



