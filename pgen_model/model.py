import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        n_atcs,
        n_drugs,
        n_genotypes,
        n_genes,
        n_alleles,
        
        
        # --- Dimensiones ---
        embedding_dim,
        # --- Arquitectura ---
        n_layers, #### AÑADIDO TESTEO
        
        hidden_dim,
        dropout_rate,
        # --- Clases únicas Outputs (¡EL CAMBIO!) ---
        # Un diccionario que mapea nombre_del_target -> n_clases
        target_dims: dict,
    ):
        super().__init__()

        self.n_fields = 5  # Drug, Gene, Allele, Genotype

        
        
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        # --- 1. Capas de Embedding (Igual) ---
        self.atc_emb = nn.Embedding(n_atcs, embedding_dim)
        self.drug_emb = nn.Embedding(n_drugs, embedding_dim)
        self.geno_emb = nn.Embedding(n_genotypes, embedding_dim)
        self.gene_emb = nn.Embedding(n_genes, embedding_dim)
        self.allele_emb = nn.Embedding(n_alleles, embedding_dim)

        # --- 2. Rama "Deep" (Igual) ---
        deep_input_dim = self.n_fields * embedding_dim
        """
        self.fc1 = nn.Linear(deep_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        """ #AÑADIDO TESTEO
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        #deep_output_dim = hidden_dim // 4
        
        #AÑADIDO TESTEO
        self.deep_layers = nn.ModuleList()
        self.deep_layers.append(nn.Linear(deep_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.deep_layers.append(nn.Linear(hidden_dim, hidden_dim))
        deep_output_dim = hidden_dim
        
        

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

    def forward(self, atc, drug, genotype, gene, allele):

        # --- 1. Obtener Embeddings  ---
        atc_vec = self.atc_emb(atc)
        drug_vec = self.drug_emb(drug)
        geno_vec = self.geno_emb(genotype)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)

        # --- 2. CÁLCULO RAMA "DEEP"  ---
        deep_input = torch.cat([atc_vec, drug_vec, geno_vec, gene_vec, allele_vec], dim=-1)
        '''"""
        deep_x = self.gelu(self.fc1(deep_input))
        deep_x = self.dropout(deep_x)
        deep_x = self.gelu(self.fc2(deep_x))
        deep_x = self.dropout(deep_x)
        deep_output = self.gelu(self.fc3(deep_x))
        deep_output = self.dropout(deep_output)
        """''' #AÑADIDO TESTEO
        deep_x = deep_input
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
            deep_x = self.gelu(deep_x)
            deep_x = self.dropout(deep_x)
            
        deep_output = deep_x
        

        # --- 3. CÁLCULO RAMA "FM"  ---
        embeddings = [atc_vec, drug_vec, geno_vec, gene_vec, allele_vec]
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
            
            if task_priorities is not None:
                priority = task_priorities.get(task_name, 1.0)
                prioritized_loss = loss_value * priority
                weighted_task_loss = (weight * prioritized_loss) + s_t
            
            else:
                weighted_task_loss = (weight * loss_value) + s_t
            
            
            
            
            weighted_loss_total += weighted_task_loss

        # Devuelve la suma de todas las pérdidas de tareas ponderadas
        return weighted_loss_total  #type: ignore 
    
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
        n_layers, #### AÑADIDO TESTEO
        
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
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        #deep_output_dim = hidden_dim // 4
        
        #AÑADIDO TESTEO
        self.deep_layers = nn.ModuleList()
        self.deep_layers.append(nn.Linear(deep_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.deep_layers.append(nn.Linear(hidden_dim, hidden_dim))
        deep_output_dim = hidden_dim

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
        """
        deep_x = self.gelu(self.fc1(deep_input))
        deep_x = self.dropout(deep_x)
        deep_x = self.gelu(self.fc2(deep_x))
        deep_x = self.dropout(deep_x)
        deep_output = self.gelu(self.fc3(deep_x))
        deep_output = self.dropout(deep_output)
        """ #AÑADIDO TESTEO
        deep_x = deep_input
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
            deep_x = self.gelu(deep_x)
            deep_x = self.dropout(deep_x)
            
        deep_output = deep_x
        

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
        """"
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
            
            if task_priorities is not None:
                priority = task_priorities.get(task_name, 1.0)
                prioritized_loss = loss_value * priority
                weighted_task_loss = (weight * prioritized_loss) + s_t
            
            else:
                weighted_task_loss = (weight * loss_value) + s_t
            
            weighted_loss_total += weighted_task_loss

        # Devuelve la suma de todas las pérdidas de tareas ponderadas
        return weighted_loss_total  #type: ignore 
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
        n_layers, #### AÑADIDO TESTEO
        
        # --- Arquitectura ---
        hidden_dim,
        dropout_rate,
        
        # --- Clases únicas Outputs (¡EL CAMBIO!) ---
        # Un diccionario que mapea nombre_del_target -> n_clases
        target_dims: dict,
    ):
        super().__init__()

        self.n_fields = 4  # Drug, Gene, Allele, Genotype
        
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        # --- 1. Capas de Embedding ---
        self.drug_emb = nn.Embedding(n_drugs, embedding_dim)
        self.geno_emb = nn.Embedding(n_genotypes, embedding_dim)
        self.gene_emb = nn.Embedding(n_genes, embedding_dim)
        self.allele_emb = nn.Embedding(n_alleles, embedding_dim)

        # --- 2. Rama "Deep" (Igual) ---
        deep_input_dim = self.n_fields * embedding_dim
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        self.attention_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=self.n_fields, 
            dim_feedforward=hidden_dim, # Usa tu hidden_dim
            dropout=dropout_rate, 
            batch_first=True, # ¡Muy importante!
        )
        
        self.deep_layers = nn.ModuleList()
        self.deep_layers.append(nn.Linear(deep_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.deep_layers.append(nn.Linear(hidden_dim, hidden_dim))
        deep_output_dim = hidden_dim

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

        emb_stack = torch.stack(
            [drug_vec, geno_vec, gene_vec, allele_vec], 
            dim=1
        )
        
        # --- 2.1 Aplicar Atención ---
        output_attn = self.attention_layer(emb_stack)
        deep_input = output_attn.flatten(start_dim=1)
        deep_x = deep_input
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
            deep_x = self.gelu(deep_x)
            deep_x = self.dropout(deep_x)
            
        deep_output = deep_x
        

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
        """"
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
            
            if task_priorities is not None:
                priority = task_priorities.get(task_name, 1.0)
                prioritized_loss = loss_value * priority
                weighted_task_loss = (weight * prioritized_loss) + s_t
            
            else:
                weighted_task_loss = (weight * loss_value) + s_t
            
            weighted_loss_total += weighted_task_loss

        # Devuelve la suma de todas las pérdidas de tareas ponderadas
        return weighted_loss_total  #type: ignore 
    '''