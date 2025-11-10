from typing import Dict, Tuple
import itertools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
        attention_dim_feedforward: int | None = None,
        attention_dropout: float = 0.1,
        num_attention_layers: int = 1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        activation_function: str = "gelu",
        fm_dropout: float = 0.0,
        fm_hidden_layers: int = 0,
        fm_hidden_dim: int = 256,
        embedding_dropout: float = 0.0,
        separate_embedding_dims: Dict[str, int] | None = None,
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
        if separate_embedding_dims is not None:
            drug_dim = separate_embedding_dims.get("drug", embedding_dim)
            genalle_dim = separate_embedding_dims.get("genalle", embedding_dim)
            gene_dim = separate_embedding_dims.get("gene", embedding_dim)
            allele_dim = separate_embedding_dims.get("allele", embedding_dim)
        else:
            drug_dim = genalle_dim = gene_dim = allele_dim = embedding_dim
            
        self.drug_emb = nn.Embedding(n_drugs, drug_dim)
        self.genal_emb = nn.Embedding(n_genalles, genalle_dim)
        self.gene_emb = nn.Embedding(n_genes, gene_dim)
        self.allele_emb = nn.Embedding(n_alleles, allele_dim)
            

        # 2. Activation function
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.activation_function = activation_function.lower()
        if self.activation_function == "gelu":
            self.activation = nn.GELU()
        elif self.activation_function == "relu":
            self.activation = nn.ReLU()
        elif self.activation_function == "swish":
            self.activation = nn.SiLU()  # Swish = SiLU in PyTorch
        elif self.activation_function == "mish":
            self.activation = nn.Mish()
        else:
            self.activation = nn.GELU()  # Default fallback
        
        
        # 3. Normalization layers
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        if use_layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])


        # 2. Deep branch with attention
        deep_input_dim = drug_dim + genalle_dim + gene_dim + allele_dim
        self.dropout = nn.Dropout(dropout_rate)
        
        # Configure attention heads
        nhead = self._get_valid_nhead(embedding_dim)
        logger.debug(f"Using {nhead} attention heads for embedding_dim={embedding_dim}")
        
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=nhead,
                dim_feedforward=attention_dim_feedforward if attention_dim_feedforward else hidden_dim,
                dropout=attention_dropout,
                batch_first=True,
            ) for _ in range(num_attention_layers)
        ])
        
        # Deep layers
        self.deep_layers = nn.ModuleList()
        self.deep_layers.append(nn.Linear(deep_input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.deep_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 3. FM branch
        self.fm_dropout = nn.Dropout(fm_dropout)
        fm_interaction_dim = (self.n_fields * (self.n_fields - 1)) // 2
        
        if fm_hidden_layers > 0:
            self.fm_layers = nn.ModuleList()
            self.fm_layers.append(nn.Linear(fm_interaction_dim, fm_hidden_dim))
            for _ in range(fm_hidden_layers - 1):
                self.fm_layers.append(nn.Linear(fm_hidden_dim, fm_hidden_dim))
            fm_output_dim = fm_hidden_dim
        else:
            self.fm_layers = None
            fm_output_dim = fm_interaction_dim

        # 4. Combined output dimension
        combined_dim = hidden_dim + fm_output_dim
        
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
        drug_vec = self.embedding_dropout(self.drug_emb(drug))
        genal_vec = self.embedding_dropout(self.genal_emb(genalle))
        gene_vec = self.embedding_dropout(self.gene_emb(gene))
        allele_vec = self.embedding_dropout(self.allele_emb(allele))

        # 2. Deep branch with attention
        emb_stack = torch.stack(
            [drug_vec, genal_vec, gene_vec, allele_vec],
            dim=1
        )
        
        # Apply attention
        attention_output = emb_stack
        for attention_layer in self.attention_layers:
            attention_output = attention_layer(attention_output)
        
        deep_input = attention_output.flatten(start_dim=1)
        
        # Pass through deep layers
        deep_x = deep_input
        for i, layer in enumerate(self.deep_layers):
            deep_x = layer(deep_x)
            
            # Apply normalization if specified
            if self.use_batch_norm and i < len(self.batch_norms):
                deep_x = self.batch_norms[i](deep_x)
            if self.use_layer_norm and i < len(self.layer_norms):
                deep_x = self.layer_norms[i](deep_x)
            
            deep_x = self.activation(deep_x)
            deep_x = self.dropout(deep_x)
        
        deep_output = deep_x
        

        # 3. FM branch
        embeddings = [drug_vec, genal_vec, gene_vec, allele_vec]
        fm_outputs = []
        for emb_i, emb_j in itertools.combinations(embeddings, 2):
            dot_product = torch.sum(emb_i * emb_j, dim=-1, keepdim=True)
            fm_outputs.append(dot_product)
        
        fm_output = torch.cat(fm_outputs, dim=-1)
        fm_output = self.fm_dropout(fm_output)
        
        if self.fm_layers is not None:
            for fm_layer in self.fm_layers:
                fm_output = fm_layer(fm_output)
                fm_output = self.activation(fm_output)
                fm_output = self.fm_dropout(fm_output)
                
                

        # 4. Combine branches
        combined_vec = torch.cat([deep_x, fm_output], dim=-1)

        # 5. Multi-task predictions
        predictions = {}
        for name, head_layer in self.output_heads.items():
            predictions[name] = head_layer(combined_vec)
        
        return predictions

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
