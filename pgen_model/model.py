'''
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PGenModel(nn.Module):
    """
    Modelo adaptable para entradas separadas de Drug y Genotype.
    """
    def __init__(self, n_drugs, n_genotypes, emb_dim_drug, emb_dim_geno, hidden_dim, dropout_rate, n_outputs):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim_drug)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim_geno)
        # El input total es la suma de ambas dimensiones de embedding
        self.fc1 = nn.Linear(emb_dim_drug + emb_dim_geno, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_dim // 2, n_outputs)

    def forward(self, drug, genotype):
        drug_vec = self.drug_emb(drug)
        geno_vec = self.geno_emb(genotype)
        # Concatenar embeddings (dim=-1 asegura que si están en batch, se concatene a nivel de feature)
        x = torch.cat([drug_vec, geno_vec], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        out = self.output(x)
        return out
''''''
import torch
import torch.nn as nn


import torch
import torch.nn as nn

class PGenModel(nn.Module):
    """
    Modelo para entradas Drug, Gene, Allele, Genotype y salidas separadas Outcome, Variation, Effect.
    Arquitectura con 3 capas (2 ocultas) y activación GELU para mejor rendimiento en problemas complejos.
    """
    def __init__(
        self, n_drugs, n_genes, n_alleles, n_genotypes,
        emb_dim_drug, emb_dim_gene, emb_dim_allele, emb_dim_geno,
        hidden_dim, dropout_rate,
        n_outcomes, n_variations, n_effects
    ):
        super().__init__()
        self.drug_emb = nn.Embedding(n_drugs, emb_dim_drug)
        self.gene_emb = nn.Embedding(n_genes, emb_dim_gene)
        self.allele_emb = nn.Embedding(n_alleles, emb_dim_allele)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim_geno)
        emb_total = emb_dim_drug + emb_dim_gene + emb_dim_allele + emb_dim_geno
        
        self.fc1 = nn.Linear(emb_total, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        # Salidas separadas
        self.outcome_head = nn.Linear(hidden_dim, n_outcomes)
        self.variation_head = nn.Linear(hidden_dim, n_variations)
        self.effect_head = nn.Linear(hidden_dim, n_effects)

    def forward(self, drug, gene, allele, genotype):
        drug_vec = self.drug_emb(drug)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)
        geno_vec = self.geno_emb(genotype)
        x = torch.cat([drug_vec, gene_vec, allele_vec, geno_vec], dim=-1)
        x = self.gelu(self.fc1(x))
        x = self.dropout(x)
        x = self.gelu(self.fc2(x))
        x = self.dropout(x)
        x = self.gelu(self.fc3(x))
        x = self.dropout(x)
        outcome = self.outcome_head(x)
        variation = self.variation_head(x)
        effect = self.effect_head(x)
        return outcome, variation, effect
'''
'''
import torch
import torch.nn as nn

class PGenModel(nn.Module):
    """
    Modelo para entradas Drug, Gene, Allele, Genotype y salidas separadas Outcome, Variation, Effect.
    Arquitectura optimizada para multitarea con salidas de 8, 98, y 2431 clases.
    """

    def __init__(
        self,
        n_drugs,
        n_genes,
        n_alleles,
        n_genotypes,
        emb_dim_drug,
        emb_dim_gene,
        emb_dim_allele,
        emb_dim_geno,
        hidden_dim,
        dropout_rate,
        n_outcomes,
        n_variations,
        n_effects,
    ):
        super().__init__()
        # Embeddings para cada variable categórica
        self.drug_emb = nn.Embedding(n_drugs, emb_dim_drug)
        self.gene_emb = nn.Embedding(n_genes, emb_dim_gene)
        self.allele_emb = nn.Embedding(n_alleles, emb_dim_allele)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim_geno)
        emb_total = emb_dim_drug + emb_dim_gene + emb_dim_allele + emb_dim_geno

        # Capas ocultas profundas y anchas
        self.fc1 = nn.Linear(emb_total, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc3 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Heads de salida para multitarea
        self.outcome_head = nn.Linear(hidden_dim//4, n_outcomes)
        self.variation_head = nn.Linear(512, n_variations)
        self.effect_head_extra = nn.Linear(512, 1024)
        self.effect_head = nn.Linear(1024, n_effects)

    def forward(self, drug, gene, allele, genotype):
        drug_vec = self.drug_emb(drug)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)
        geno_vec = self.geno_emb(genotype)
        x = torch.cat([drug_vec, gene_vec, allele_vec, geno_vec], dim=-1)
        x = self.gelu(self.fc1(x)); x = self.dropout(x)
        x = self.gelu(self.fc2(x)); x = self.dropout(x)
        x = self.gelu(self.fc3(x)); x = self.dropout(x)
        outcome = self.outcome_head(x)
        variation = self.variation_head(x)
        # Head especial para Effect
        effect_x = self.gelu(self.effect_head_extra(x))
        effect = self.effect_head(effect_x)
        return outcome, variation, effect
'''

import torch
import torch.nn as nn

class PGenModel(nn.Module):
    """
    Modelo multitarea para predecir Outcome, Effect_direction, Effect_category, Entity, Entity_name, Therapeutic_Outcome
    a partir de Drug, Gene, Allele y Genotype.
    """
    def __init__(
        self,
        n_drugs,
        n_genes,
        n_alleles,
        n_genotypes,
        emb_dim_drug,
        emb_dim_gene,
        emb_dim_allele,
        emb_dim_geno,
        hidden_dim,
        dropout_rate,
        n_outcomes,
        n_effect_direction,
        n_effect_category,
        n_entities,
        n_entity_names,
        n_therapeutic_outcomes
    ):
        super().__init__()
        # Embeddings para cada variable categórica input
        self.drug_emb = nn.Embedding(n_drugs, emb_dim_drug)
        self.gene_emb = nn.Embedding(n_genes, emb_dim_gene)
        self.allele_emb = nn.Embedding(n_alleles, emb_dim_allele)
        self.geno_emb = nn.Embedding(n_genotypes, emb_dim_geno)
        emb_total = emb_dim_drug + emb_dim_gene + emb_dim_allele + emb_dim_geno

        # Capas ocultas
        self.fc1 = nn.Linear(emb_total, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

        # Heads de salida (multitarea)
        self.outcome_head = nn.Linear(hidden_dim // 4, n_outcomes)
        self.effect_direction_head = nn.Linear(hidden_dim // 4, n_effect_direction)
        self.effect_category_head = nn.Linear(hidden_dim // 4, n_effect_category)
        self.entity_head = nn.Linear(hidden_dim // 4, n_entities)
        self.entity_name_head = nn.Linear(hidden_dim // 4, n_entity_names)
        self.therapeutic_outcome_head = nn.Linear(hidden_dim // 4, n_therapeutic_outcomes)

    def forward(self, drug, gene, allele, genotype):
        x = torch.cat([
            self.drug_emb(drug),
            self.gene_emb(gene),
            self.allele_emb(allele),
            self.geno_emb(genotype)
        ], dim=-1)
        x = self.gelu(self.fc1(x)); x = self.dropout(x)
        x = self.gelu(self.fc2(x)); x = self.dropout(x)
        x = self.gelu(self.fc3(x)); x = self.dropout(x)

        outcome = self.outcome_head(x)
        effect_direction = self.effect_direction_head(x)
        effect_category = self.effect_category_head(x)
        entity = self.entity_head(x)
        entity_name = self.entity_name_head(x)
        therapeutic_outcome = self.therapeutic_outcome_head(x)

        return outcome, effect_direction, effect_category, entity, entity_name, therapeutic_outcome