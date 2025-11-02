import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
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
'''


class DeepFM_PGenModel(nn.Module):
    """
    Versión generalizada del modelo DeepFM que crea
    dinámicamente los "heads" de salida basados en un
    diccionario de configuración.
    """

    def __init__(
        self,
        # --- Vocabulario (Inputs) ---
        n_drugs,
        n_genotypes,
        n_genes,
        n_alleles,
        
        
        # --- Dimensiones ---
        embedding_dim,
        # --- Arquitectura ---
        hidden_dim,
        dropout_rate,
        # --- Clases únicas Outputs (¡EL CAMBIO!) ---
        # Un diccionario que mapea nombre_del_target -> n_clases
        target_dims: dict,
    ):
        super().__init__()

        self.n_fields = 4  # Drug, Gene, Allele, Genotype

        """Testeos Cambio de Ponderaciones"""
        
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        # --- 1. Capas de Embedding (Igual) ---
        self.drug_emb = nn.Embedding(n_drugs, embedding_dim)
        self.geno_emb = nn.Embedding(n_genotypes, embedding_dim)
        self.gene_emb = nn.Embedding(n_genes, embedding_dim)
        self.allele_emb = nn.Embedding(n_alleles, embedding_dim)

        # --- 2. Rama "Deep" (Igual) ---
        deep_input_dim = self.n_fields * embedding_dim
        self.fc1 = nn.Linear(deep_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        deep_output_dim = hidden_dim // 4

        # --- 3. Rama "FM" (Igual) ---
        fm_output_dim = (self.n_fields * (self.n_fields - 1)) // 2

        # --- 4. Capa de Salida Combinada (Igual) ---
        combined_dim = deep_output_dim + fm_output_dim

        # --- 5. Heads de Salida (Dinámicos) ---
        # ModuleDict para almacenar las capas de salida
        self.output_heads = nn.ModuleDict()

        # Iteramos sobre el diccionario de configuración
        for target_name, n_classes in target_dims.items():
            # Creamos una capa lineal para cada target y la guardamos
            # en el ModuleDict usando su nombre como clave.
            self.output_heads[target_name] = nn.Linear(combined_dim, n_classes)

    def forward(self, drug, genotype, gene, allele):

        # --- 1. Obtener Embeddings  ---
        drug_vec = self.drug_emb(drug)
        geno_vec = self.geno_emb(genotype)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)

        # --- 2. CÁLCULO RAMA "DEEP"  ---
        deep_input = torch.cat([drug_vec, geno_vec, gene_vec, allele_vec], dim=-1)
        deep_x = self.gelu(self.fc1(deep_input))
        deep_x = self.dropout(deep_x)
        deep_x = self.gelu(self.fc2(deep_x))
        deep_x = self.dropout(deep_x)
        deep_output = self.gelu(self.fc3(deep_x))
        deep_output = self.dropout(deep_output)

        # --- 3. CÁLCULO RAMA "FM"  ---
        embeddings = [drug_vec, geno_vec, gene_vec, allele_vec]
        fm_outputs = []
        for emb_i, emb_j in itertools.combinations(embeddings, 2):
            dot_product = torch.sum(emb_i * emb_j, dim=-1, keepdim=True)
            fm_outputs.append(dot_product)
        fm_output = torch.cat(fm_outputs, dim=-1)

        # --- 4. COMBINACIÓN  ---
        combined_vec = torch.cat([deep_output, fm_output], dim=-1)

        # --- 5. PREDICCIONES (Dinámicas) ---
        # Diccionario de predicciones
        predictions = {}

        # Iteramos sobre nuestro ModuleDict
        for name, head_layer in self.output_heads.items():
            # Aplicamos la capa correspondiente y guardamos
            # la predicción en el diccionario de salida.
            predictions[name] = head_layer(combined_vec)

        # Devolvemos el diccionario de predicciones
        return predictions

    def calculate_weighted_loss(self, unweighted_losses: dict, task_priorities: dict) -> torch.Tensor:
        """
        Calcula la pérdida total ponderada por Incertidumbre (Uncertainty Weighting).
        Fórmula para clasificación: L_total = Σ [ L_i * exp(-s_i) + s_i ]
        donde s_i = log(σ_i²) es el parámetro aprendible.

        Args:
            unweighted_losses: Diccionario de pérdidas no ponderadas (L_i).
                               Ej: {"outcome": tensor(0.5), "effect": tensor(0.2)}

        Returns:
            La pérdida total (torch.Tensor escalar) lista para .backward().
        """
        weighted_loss_total = 0.0

        for task_name, loss_value in unweighted_losses.items():
            # 1. Obtener el parámetro aprendible 's_i' (log(sigma^2))
            s_t = self.log_sigmas[task_name]

            # 2. Calcular el peso dinámico (precision = 1/sigma^2 = exp(-s_t))
            weight = torch.exp(-s_t)

            # Establecimiento de prioridades para los pesos.
            '''
            if task_priorities is not None:
                priority = task_priorities.get(task_name, 1.0)
                prioritized_loss = loss_value * priority
                weighted_task_loss = (weight * prioritized_loss) + s_t
            
            else:
                weighted_task_loss = (weight * loss_value) + s_t
            '''
            
            priority = task_priorities.get(task_name, 1.0)
            prioritized_loss = loss_value * priority
            weighted_task_loss = (weight * prioritized_loss) + s_t
            
            weighted_loss_total += weighted_task_loss

        # Devuelve la suma de todas las pérdidas de tareas ponderadas
        return weighted_loss_total  #type: ignore 
