from typing import Dict, Tuple
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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
        n_genalles,
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

        self.n_fields = 5  # Drug, Gene, Allele, genalle

        
        
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        # --- 1. Capas de Embedding (Igual) ---
        self.atc_emb = nn.Embedding(n_atcs, embedding_dim)
        self.drug_emb = nn.Embedding(n_drugs, embedding_dim)
        self.genal_emb = nn.Embedding(n_genalles, embedding_dim)
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

    def forward(self, atc, drug, genalle, gene, allele):

        # --- 1. Obtener Embeddings  ---
        atc_vec = self.atc_emb(atc)
        drug_vec = self.drug_emb(drug)
        genal_vec = self.genal_emb(genalle)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)

        # --- 2. CÁLCULO RAMA "DEEP"  ---
        deep_input = torch.cat([atc_vec, drug_vec, genal_vec, gene_vec, allele_vec], dim=-1)
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
        embeddings = [atc_vec, drug_vec, genal_vec, gene_vec, allele_vec]
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
        n_genalles,
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

        self.n_fields = 4  # Drug, Gene, Allele, genalle

        """Testeos Cambio de Ponderaciones"""
        
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        # --- 1. Capas de Embedding (Igual) ---
        self.drug_emb = nn.Embedding(n_drugs, embedding_dim)
        self.genal_emb = nn.Embedding(n_genalles, embedding_dim)
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

    def forward(self, drug, genalle, gene, allele):

        # --- 1. Obtener Embeddings  ---
        drug_vec = self.drug_emb(drug)
        genal_vec = self.genal_emb(genalle)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)

        # --- 2. CÁLCULO RAMA "DEEP"  ---
        deep_input = torch.cat([drug_vec, genal_vec, gene_vec, allele_vec], dim=-1)
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
        embeddings = [drug_vec, genal_vec, gene_vec, allele_vec]
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
    Generalized DeepFM model for multi-task pharmacogenomics prediction.
    
    Combines Deep learning and Factorization Machine branches with multi-head
    output for simultaneous prediction of multiple targets. Uses uncertainty
    weighting for automatic task balancing.
    
    Architecture:
        - Embedding layer: Converts categorical inputs to dense vectors
        - Deep branch: Multi-layer perceptron with attention mechanism
        - FM branch: Factorization Machine for feature interactions
        - Output heads: Task-specific prediction layers
    
    References:
        - DeepFM: He et al., 2017
        - Uncertainty Weighting: Kendall & Gal, 2017
    """
    
    # Class constants for architecture
    N_FIELDS = 4  # Drug, Gene, Allele, Genalle
    VALID_NHEADS = [8, 4, 2, 1]  # Valid attention head options
    MIN_DROPOUT = 0.0
    MAX_DROPOUT = 0.9

    def __init__(
        self,
        n_drugs: int,
        n_genalles: int,
        n_genes: int,
        n_alleles: int,
        embedding_dim: int,
        n_layers: int,
        hidden_dim: int,
        dropout_rate: float,
        target_dims: Dict[str, int],
    ) -> None:
        """
        Initialize DeepFM_PGenModel.
        
        Args:
            n_drugs: Number of unique drugs in vocabulary
            n_genalles: Number of unique genotypes/alleles combinations
            n_genes: Number of unique genes
            n_alleles: Number of unique alleles
            embedding_dim: Dimension of embedding vectors
            n_layers: Number of layers in deep branch
            hidden_dim: Hidden dimension for deep layers
            dropout_rate: Dropout probability (0.0 to 0.9)
            target_dims: Dictionary mapping target names to number of classes
                        Example: {"outcome": 3, "effect_type": 5}
        
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If attention head configuration fails
        """
        super().__init__()
        
        # Validate inputs
        self._validate_inputs(
            n_drugs, n_genalles, n_genes, n_alleles,
            embedding_dim, n_layers, hidden_dim, dropout_rate, target_dims
        )
        
        self.n_fields = self.N_FIELDS
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.target_dims = target_dims
        
        # Initialize uncertainty weighting parameters
        self.log_sigmas = nn.ParameterDict()
        for target_name in target_dims.keys():
            self.log_sigmas[target_name] = nn.Parameter(
                torch.tensor(0.0, requires_grad=True)
            )
        
        # 1. Embedding layers
        self.drug_emb = nn.Embedding(n_drugs, embedding_dim)
        self.genal_emb = nn.Embedding(n_genalles, embedding_dim)
        self.gene_emb = nn.Embedding(n_genes, embedding_dim)
        self.allele_emb = nn.Embedding(n_alleles, embedding_dim)

        # 2. Deep branch with attention
        deep_input_dim = self.n_fields * embedding_dim
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # Configure attention heads
        nhead = self._get_valid_nhead(embedding_dim)
        logger.debug(f"Using {nhead} attention heads for embedding_dim={embedding_dim}")
        
        self.attention_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            batch_first=True,
        )
        
        # Deep layers
        self.deep_layers = nn.ModuleList()
        self.deep_layers.append(nn.Linear(deep_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.deep_layers.append(nn.Linear(hidden_dim, hidden_dim))
        deep_output_dim = hidden_dim

        # 3. FM branch
        fm_output_dim = (self.n_fields * (self.n_fields - 1)) // 2

        # 4. Combined output dimension
        combined_dim = deep_output_dim + fm_output_dim

        # 5. Multi-task output heads
        self.output_heads = nn.ModuleDict()
        for target_name, n_classes in target_dims.items():
            self.output_heads[target_name] = nn.Linear(combined_dim, n_classes)
        
        logger.info(f"DeepFM_PGenModel initialized with {len(self.output_heads)} output heads")
    
    @staticmethod
    def _validate_inputs(
        n_drugs: int, n_genalles: int, n_genes: int, n_alleles: int,
        embedding_dim: int, n_layers: int, hidden_dim: int,
        dropout_rate: float, target_dims: Dict[str, int]
    ) -> None:
        """Validate all input parameters."""
        if n_drugs <= 0 or n_genalles <= 0 or n_genes <= 0 or n_alleles <= 0:
            raise ValueError("All vocabulary sizes must be positive integers")
        
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        
        if not (0.0 <= dropout_rate <= 0.9):
            raise ValueError(f"dropout_rate must be in [0.0, 0.9], got {dropout_rate}")
        
        if not target_dims or not isinstance(target_dims, dict):
            raise ValueError("target_dims must be a non-empty dictionary")
        
        for target_name, n_classes in target_dims.items():
            if n_classes <= 0:
                raise ValueError(f"target_dims['{target_name}'] must be positive, got {n_classes}")
    
    @staticmethod
    def _get_valid_nhead(embedding_dim: int) -> int:
        """
        Get the largest valid number of attention heads for given embedding dimension.
        
        Args:
            embedding_dim: Embedding dimension
            
        Returns:
            Valid number of attention heads
            
        Raises:
            RuntimeError: If no valid nhead found
        """
        valid_nheads = [h for h in DeepFM_PGenModel.VALID_NHEADS if embedding_dim % h == 0]
        if not valid_nheads:
            raise RuntimeError(
                f"No valid attention heads for embedding_dim={embedding_dim}. "
                f"embedding_dim must be divisible by one of {DeepFM_PGenModel.VALID_NHEADS}"
            )
        return valid_nheads[0]

    def forward(
        self,
        drug: torch.Tensor,
        genalle: torch.Tensor,
        gene: torch.Tensor,
        allele: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            drug: Tensor of drug indices, shape (batch_size,)
            genalle: Tensor of genotype indices, shape (batch_size,)
            gene: Tensor of gene indices, shape (batch_size,)
            allele: Tensor of allele indices, shape (batch_size,)
        
        Returns:
            Dictionary mapping target names to prediction tensors.
            Each tensor has shape (batch_size, n_classes) for that target.
        
        Raises:
            RuntimeError: If input tensors have incompatible shapes
        """
        # Validate input shapes
        batch_size = drug.shape[0]
        if not all(t.shape[0] == batch_size for t in [genalle, gene, allele]):
            raise RuntimeError("All input tensors must have the same batch size")

        # 1. Get embeddings
        drug_vec = self.drug_emb(drug)
        genal_vec = self.genal_emb(genalle)
        gene_vec = self.gene_emb(gene)
        allele_vec = self.allele_emb(allele)

        # 2. Deep branch with attention
        emb_stack = torch.stack(
            [drug_vec, genal_vec, gene_vec, allele_vec],
            dim=1
        )
        
        # Apply attention
        output_attn = self.attention_layer(emb_stack)
        deep_input = output_attn.flatten(start_dim=1)
        
        # Pass through deep layers
        deep_x = deep_input
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
            deep_x = self.gelu(deep_x)
            deep_x = self.dropout(deep_x)
        
        deep_output = deep_x

        # 3. FM branch
        embeddings = [drug_vec, genal_vec, gene_vec, allele_vec]
        fm_outputs = []
        for emb_i, emb_j in itertools.combinations(embeddings, 2):
            dot_product = torch.sum(emb_i * emb_j, dim=-1, keepdim=True)
            fm_outputs.append(dot_product)
        fm_output = torch.cat(fm_outputs, dim=-1)

        # 4. Combine branches
        combined_vec = torch.cat([deep_output, fm_output], dim=-1)

        # 5. Multi-task predictions
        predictions = {}
        for name, head_layer in self.output_heads.items():
            predictions[name] = head_layer(combined_vec)

        return predictions

    '''
    def calculate_weighted_loss(
        self,
        unweighted_losses: Dict[str, torch.Tensor],
        task_priorities: Dict[str, float],
    ) -> torch.Tensor:
        """
        Calculate total weighted loss using uncertainty weighting.
        
        Implements multi-task learning with automatic task balancing using
        learnable uncertainty parameters. Formula for classification:
        L_total = Σ [ L_i * exp(-s_i) + s_i ]
        where s_i = log(σ_i²) is the learnable parameter for task i.
        
        Args:
            unweighted_losses: Dictionary mapping task names to loss tensors.
                              Example: {"outcome": tensor(0.5), "effect_type": tensor(0.2)}
            task_priorities: Dictionary mapping task names to priority weights (optional).
                            If None, all tasks weighted equally.
                            Example: {"outcome": 1.5, "effect_type": 1.0}
        
        Returns:
            Scalar tensor representing total weighted loss, ready for .backward()
        
        Raises:
            KeyError: If a task in unweighted_losses was not in target_dims during init
        """
        weighted_loss_total = 0.0

        for task_name, loss_value in unweighted_losses.items():
            # Validate task exists
            if task_name not in self.log_sigmas:
                raise KeyError(
                    f"Task '{task_name}' not found in model. "
                    f"Available tasks: {list(self.log_sigmas.keys())}"
                )
            
            # Get learnable uncertainty parameter
            s_t = self.log_sigmas[task_name]

            # Calculate dynamic weight (precision = 1/sigma^2 = exp(-s_t))
            weight = torch.exp(-s_t)

            # Apply task priorities if provided
            if task_priorities is not None and task_name in task_priorities:
                priority = task_priorities[task_name]
                prioritized_loss = loss_value * priority
                weighted_task_loss = (weight * prioritized_loss) + s_t
            else:
                weighted_task_loss = (weight * loss_value) + s_t
            
            weighted_loss_total += weighted_task_loss

        return weighted_loss_total # type: ignore
    '''
    def __repr__(self) -> str:
        """String representation for debugging and logging."""
        return (
            f"DeepFM_PGenModel("
            f"embedding_dim={self.embedding_dim}, "
            f"n_layers={self.n_layers}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dropout_rate={self.dropout_rate}, "
            f"n_tasks={len(self.output_heads)}, "
            f"tasks={list(self.target_dims.keys())})"
        )
