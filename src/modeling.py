# Pharmagen - Modeling
# Copyright (C) 2025 Adrim Hamed Outmani

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

@dataclass(frozen=True)
class ModelConfig:
    """
    Immutable configuration object for the Pharmagen Model.
    Acts as the single source of truth for hyperparameters.

    Args:
        n_features (dict[str, int]): Mapping of feature names to their cardinality (vocab size).
        target_dims (dict[str, int]): Mapping of target names to their output dimension.
        embedding_dim (int): Dimension of the dense embedding space.
        hidden_dim (int): Dimension of the hidden layers in the MLP.
        use_transformer (bool): Whether to include the Transformer encoder.

    Example:
        >>> cfg = ModelConfig(n_features={'drug': 100}, target_dims={'y': 1})
        >>> cfg.embedding_dim
        64
    """
    # Data Dimensions
    n_features: dict[str, int]
    target_dims: dict[str, int]

    # Embedding
    embedding_dim: int = 64
    embedding_dropout: float = 0.1

    # Deep Component (MLP)
    hidden_dim: int = 256
    n_layers: int = 3
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    activation: str = "gelu"

    # Transformer Component
    use_transformer: bool = True
    attn_dim_feedforward: int = 1024
    attn_dropout: float = 0.1
    num_attn_layers: int = 2
    attn_heads: int = 4

    # FM Component
    fm_hidden_layers: int = 1
    fm_hidden_dim: int = 128
    fm_dropout: float = 0.1

class MLP(nn.Module):
    """
    Generic Multi-Layer Perceptron generator.
    
    Constructs a sequence of Linear -> [Norm] -> Activation -> Dropout.
    """
    def __init__( # noqa
        self,
        in_dim: int,
        hidden_dims: list[int],
        dropout: float,
        activation: str,
        use_bn: bool = False,
        use_ln: bool = False
    ):
        super().__init__()
        layers = []
        act_class = getattr(nn, activation.upper(), nn.GELU)

        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(h_dim))
            if use_ln:
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_class())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        self.network = nn.Sequential(*layers)
        self.output_dim = curr_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FactorizationMachine(nn.Module):
    """
    Vectorized Factorization Machine Layer (2nd order interactions).
    
    Implements the formula: 
    0.5 * ( (sum(x))^2 - sum(x^2) )
    """
    def __init__(self, reduce_sum: bool = True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates feature interactions.

        Args:
            x (Tensor): Input tensor of shape (Batch, Num_Features, Embedding_Dim).

        Returns:
            Tensor: Interaction tensor of shape (Batch, Embedding_Dim).
        """
        square_of_sum = torch.sum(x, dim=1).pow(2)
        sum_of_squares = torch.sum(x.pow(2), dim=1)
        interaction = 0.5 * (square_of_sum - sum_of_squares)
        return interaction

class PharmagenDeepFM(nn.Module):
    """
    Hybrid DeepFM Architecture extended with Transformer Encoders.
    
    Combines:
    1. Embedding Layer (Categorical Features)
    2. Transformer Encoder (Feature Cross-Attention)
    3. Deep Component (MLP)
    4. Wide Component (Factorization Machine)
    
    Args:
        config (ModelConfig): Configuration object containing architecture specs.
    
    Example:
        >>> cfg = ModelConfig(n_features={'a': 10}, target_dims={'b': 1})
        >>> model = PharmagenDeepFM(cfg)
        >>> out = model(inputs)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.cfg = config
        self.feature_names = list(config.n_features.keys())

        # 1. Embeddings
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(num_embeddings=count, embedding_dim=config.embedding_dim)
            for feat, count in config.n_features.items()
        })
        self.emb_dropout = nn.Dropout(config.embedding_dropout)

        # 2. Transformer Feature Extractor
        self.transformer = None
        if config.use_transformer:
            encoder_layer = TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.attn_heads,
                dim_feedforward=config.attn_dim_feedforward,
                dropout=config.attn_dropout,
                activation=config.activation,
                batch_first=True,
                norm_first=True
            )
            self.transformer = TransformerEncoder(encoder_layer, num_layers=config.num_attn_layers)

        # 3. Deep MLP Path
        deep_in_dim = len(self.feature_names) * config.embedding_dim
        deep_hidden_layers = [config.hidden_dim] * config.n_layers
        self.deep_mlp = MLP(
            in_dim=deep_in_dim,
            hidden_dims=deep_hidden_layers,
            dropout=config.dropout_rate,
            activation=config.activation,
            use_bn=config.use_batch_norm,
            use_ln=config.use_layer_norm
        )

        # 4. FM Path
        self.fm_layer = FactorizationMachine()
        self.fm_dropout = nn.Dropout(config.fm_dropout)

        self.fm_mlp = None
        fm_out_dim = config.embedding_dim

        if config.fm_hidden_layers > 0:
            fm_hidden_struct = [config.fm_hidden_dim] * config.fm_hidden_layers
            self.fm_mlp = MLP(
                in_dim=config.embedding_dim,
                hidden_dims=fm_hidden_struct,
                dropout=config.fm_dropout,
                activation=config.activation,
                use_bn=config.use_batch_norm
            )
            fm_out_dim = config.fm_hidden_dim

        # 5. Output Heads
        combined_dim = self.deep_mlp.output_dim + fm_out_dim
        self.heads = nn.ModuleDict({
            target: nn.Linear(combined_dim, dim)
            for target, dim in config.target_dims.items()
        })

        self._initialize_weights()

    def _initialize_weights(self):
        """Selective initialization avoiding Transformer internals."""
        for name, m in self.named_modules():
            if "transformer" in name:
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, x_cat: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x_cat (dict): Dictionary of input tensors {feature_name: (Batch,)}.
            
        Returns:
            dict: Dictionary of output logits {target_name: (Batch, Dim)}.
        """
        # 1. Embedding Lookup
        emb_list = [self.embeddings[name](x_cat[name]) for name in self.feature_names]
        emb_stack = torch.stack(emb_list, dim=1)
        emb_stack = self.emb_dropout(emb_stack)

        # 2. Deep Path
        deep_features = emb_stack
        if self.transformer:
            deep_features = self.transformer(deep_features)

        deep_flat = deep_features.flatten(start_dim=1)
        deep_out = self.deep_mlp(deep_flat)

        # 3. FM Path
        fm_out = self.fm_layer(emb_stack)
        fm_out = self.fm_dropout(fm_out)

        if self.fm_mlp:
            fm_out = self.fm_mlp(fm_out)

        # 4. Fusion
        combined = torch.cat([deep_out, fm_out], dim=-1)

        return {target: head(combined) for target, head in self.heads.items()}
