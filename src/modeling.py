# Pharmagen - Modeling
# Unified Architecture and Loss Functions.
# Adheres to SOLID: Grouping cohesive logic.

from typing import Dict, List, Any, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer

# =============================================================================
# 1. ARCHITECTURE: DEEPFM (Deep Factorization Machine)
# =============================================================================

class DeepFM_PGenModel(nn.Module):
    def __init__(
        self,
        n_features: Dict[str, int],
        target_dims: Dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        n_layers: int,
        attention_dim_feedforward: int = 2048,
        attention_dropout: float = 0.1,
        num_attention_layers: int = 1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        activation_function: str = "gelu",
        fm_dropout: float = 0.1,
        fm_hidden_layers: int = 0,
        fm_hidden_dim: int = 256,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_names = list(n_features.keys())
        self.target_names = list(target_dims.keys())
        
        # 1. Embeddings
        self.embeddings = nn.ModuleDict({
            feat: nn.Embedding(num, embedding_dim) 
            for feat, num in n_features.items()
        })
        self.emb_dropout = nn.Dropout(embedding_dropout)

        # 2. Deep Component (Transformer + MLP)
        # Transformer enforces interaction between features
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8 if embedding_dim % 8 == 0 else 4,
            dim_feedforward=attention_dim_feedforward,
            dropout=attention_dropout,
            activation=activation_function,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_attention_layers)
        
        deep_input_dim = len(n_features) * embedding_dim
        self.deep_mlp = self._make_mlp(
            deep_input_dim, hidden_dim, n_layers, dropout_rate, 
            use_batch_norm, use_layer_norm, activation_function
        )

        # 3. FM Component (Vectorized)
        self.fm_dropout = nn.Dropout(fm_dropout)
        self.fm_mlp = None
        fm_out_dim = embedding_dim
        
        if fm_hidden_layers > 0:
            self.fm_mlp = self._make_mlp(
                embedding_dim, fm_hidden_dim, fm_hidden_layers, fm_dropout,
                use_batch_norm, False, activation_function
            )
            fm_out_dim = fm_hidden_dim

        # 4. Output Heads (Multi-Task)
        combined_dim = hidden_dim + fm_out_dim
        self.heads = nn.ModuleDict({
            target: nn.Linear(combined_dim, dim) 
            for target, dim in target_dims.items()
        })
        
        self.apply(self._init_weights)

    def _make_mlp(self, in_dim, hid_dim, layers, drop, bn, ln, act):
        net = []
        act_fn = getattr(nn, act.upper(), nn.GELU)()
        curr_in = in_dim
        for _ in range(layers):
            net.append(nn.Linear(curr_in, hid_dim))
            if bn: net.append(nn.BatchNorm1d(hid_dim))
            if ln: net.append(nn.LayerNorm(hid_dim))
            net.append(act_fn)
            net.append(nn.Dropout(drop))
            curr_in = hid_dim
        return nn.Sequential(*net)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Stack embeddings: [Batch, N_Feats, Emb_Dim]
        emb_list = [self.embeddings[f](x[f]) for f in self.feature_names]
        emb_stack = torch.stack(emb_list, dim=1)
        emb_stack = self.emb_dropout(emb_stack)

        # Deep Path
        trans_out = self.transformer(emb_stack)
        deep_out = self.deep_mlp(trans_out.flatten(1))

        # FM Path (Vectorized 2nd order interactions)
        sum_sq = torch.sum(emb_stack, dim=1).pow(2)
        sq_sum = torch.sum(emb_stack.pow(2), dim=1)
        fm_out = 0.5 * (sum_sq - sq_sum)
        fm_out = self.fm_dropout(fm_out)
        if self.fm_mlp:
            fm_out = self.fm_mlp(fm_out)

        # Combine
        combined = torch.cat([deep_out, fm_out], dim=-1)
        
        return {t: head(combined) for t, head in self.heads.items()}

# =============================================================================
# 2. MODEL FACTORY
# =============================================================================

def create_model(model_name: str, n_features: Dict[str, int], target_dims: Dict[str, int], params: Dict[str, Any]) -> nn.Module:
    """
    Factory function to instantiate models based on configuration.
    Decouples architecture selection from the training pipeline.
    """

    model_kwargs = {
        k: v for k, v in params.items() 
        if k in [
            "embedding_dim", "hidden_dim", "dropout_rate", "n_layers",
            "attention_dim_feedforward", "attention_dropout", "num_attention_layers",
            "use_batch_norm", "use_layer_norm", "activation_function",
            "fm_dropout", "fm_hidden_layers", "fm_hidden_dim", "embedding_dropout"
        ]
    }

    return DeepFM_PGenModel(
        n_features=n_features,
        target_dims=target_dims,
        **model_kwargs
    )

