# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

logger = logging.getLogger(__name__)


class DeepFM_PGenModel(nn.Module):
    """
    Modelo DeepFM generalizado optimizado para Farmacogenética.
    Implementa Deep Branch (Transformer) y FM Branch (Interacciones vectorizadas).
    """

    def __init__(
        self,
        n_features: dict[str, int],
        target_dims: dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        n_layers: int,
        attention_dim_feedforward: int | None = None,
        attention_dropout: float = 0.1,
        num_attention_layers: int = 1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        activation_function: str = "gelu",
        fm_dropout: float = 0.1,
        fm_hidden_layers: int = 0,
        fm_hidden_dim: int = 256,
        embedding_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_names = list(n_features.keys())
        self.n_fields = len(n_features)
        self.embedding_dim = embedding_dim
        self.target_dims = target_dims

        # ------------------------------------------------------------------
        # 1. Embeddings Dinámicos
        # ------------------------------------------------------------------
        self.embeddings = nn.ModuleDict()
        for feat, num_classes in n_features.items():
            self.embeddings[feat] = nn.Embedding(num_classes, embedding_dim)

        self.emb_dropout = nn.Dropout(embedding_dropout)

        # ------------------------------------------------------------------
        # 2. Deep Branch (Transformer + MLP)
        # ------------------------------------------------------------------
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=self._get_valid_nhead(embedding_dim),
            dim_feedforward=attention_dim_feedforward or hidden_dim,
            dropout=attention_dropout,
            activation=activation_function,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_attention_layers
        )

        # Input dim para MLP después de aplanar: N_Fields * Emb_Dim
        deep_input_dim = self.n_fields * embedding_dim
        
        self.deep_mlp = self._build_mlp(
            input_dim=deep_input_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout_rate,
            bn=use_batch_norm,
            ln=use_layer_norm,
            act_fn=activation_function,
        )

        # ------------------------------------------------------------------
        # 3. FM Branch (Vectorizada)
        # ------------------------------------------------------------------
        # NOTA: Al vectorizar, la salida de la interacción FM tiene tamaño 'embedding_dim',
        # no N*(N-1)/2. Esto es mucho más eficiente y estable.
        self.fm_dropout = nn.Dropout(fm_dropout)
        fm_input_dim = embedding_dim 

        if fm_hidden_layers > 0:
            self.fm_mlp = self._build_mlp(
                input_dim=fm_input_dim,
                hidden_dim=fm_hidden_dim,
                n_layers=fm_hidden_layers,
                dropout=fm_dropout,
                bn=use_batch_norm,
                ln=False,
                act_fn=activation_function,
            )
            self.fm_out_dim = fm_hidden_dim
        else:
            self.fm_mlp = None
            self.fm_out_dim = fm_input_dim

        # ------------------------------------------------------------------
        # 4. Output Heads
        # ------------------------------------------------------------------
        combined_dim = hidden_dim + self.fm_out_dim
        self.heads = nn.ModuleDict()
        
        for target, dim in target_dims.items():
            self.heads[target] = nn.Linear(combined_dim, dim)

        # Inicialización de pesos personalizada (Best Practice para DL)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Inicialización robusta (Xavier/Kaiming) según el tipo de capa."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def _build_mlp(
        self, input_dim: int, hidden_dim: int, n_layers: int, 
        dropout: float, bn: bool, ln: bool, act_fn: str
    ) -> nn.Sequential:
        """Constructor de MLP simplificado y legible."""
        layers = []
        in_d = input_dim
        
        # Mapeo de activaciones
        act_map = {
            "gelu": nn.GELU, "relu": nn.ReLU, 
            "silu": nn.SiLU, "mish": nn.Mish
        }
        activation_cls = act_map.get(act_fn.lower(), nn.GELU)

        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if ln:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(activation_cls())
            layers.append(nn.Dropout(dropout))
            in_d = hidden_dim
            
        return nn.Sequential(*layers)

    @staticmethod
    def _get_valid_nhead(dim: int) -> int:
        for h in [8, 4, 2, 1]:
            if dim % h == 0:
                return h
        return 1

    def forward(
        self, inputs: dict[str, torch.Tensor], **kwargs
    ) -> dict[str, torch.Tensor]:
        
        if kwargs:
            inputs = {**inputs, **kwargs}

        # ----------------------------------------------------------
        # 1. Obtención y Apilado de Embeddings
        # ----------------------------------------------------------
        # Validación rápida de claves existentes
        missing = [f for f in self.feature_names if f not in inputs]
        if missing:
            raise ValueError(f"Features faltantes en forward: {missing}")

        # Lista de tensores: [Batch, Emb_Dim] -> Stack -> [Batch, N_Fields, Emb_Dim]
        emb_list = [self.embeddings[feat](inputs[feat]) for feat in self.feature_names]
        emb_stack = torch.stack(emb_list, dim=1)
        emb_stack = self.emb_dropout(emb_stack)

        # ----------------------------------------------------------
        # 2. Deep Branch (Transformer)
        # ----------------------------------------------------------
        # [Batch, N_Fields, Emb_Dim] -> Transformer -> [Batch, N_Fields, Emb_Dim]
        trans_out = self.transformer(emb_stack)
        
        # Flatten: [Batch, N_Fields * Emb_Dim]
        deep_in = trans_out.flatten(start_dim=1)
        deep_out = self.deep_mlp(deep_in)

        # ----------------------------------------------------------
        # 3. FM Branch (Vectorizada - O(N) Complexity)
        # ----------------------------------------------------------
        # Fórmula FM: 0.5 * ( (Sum(V))^2 - Sum(V^2) )
        # Sumamos a través de la dimensión de campos (dim=1)
        
        # A. Suma de embeddings al cuadrado: (v1 + v2 + ...)^2
        sum_of_embs = torch.sum(emb_stack, dim=1)  # -> [Batch, Emb_Dim]
        sum_sq = torch.square(sum_of_embs)
        
        # B. Suma de cuadrados de embeddings: (v1^2 + v2^2 + ...)
        sq_of_embs = torch.square(emb_stack)       # -> [Batch, N_Fields, Emb_Dim]
        sq_sum = torch.sum(sq_of_embs, dim=1)      # -> [Batch, Emb_Dim]
        
        # C. Interacción final
        fm_interaction = 0.5 * (sum_sq - sq_sum)   # -> [Batch, Emb_Dim]
        
        fm_out = self.fm_dropout(fm_interaction)

        if self.fm_mlp:
            fm_out = self.fm_mlp(fm_out)

        # ----------------------------------------------------------
        # 4. Concatenación y Salida
        # ----------------------------------------------------------
        # [Batch, Hidden_Deep + Hidden_FM]
        combined = torch.cat([deep_out, fm_out], dim=-1)

        # Generar salidas para cada tarea
        outputs = {
            task: head(combined) 
            for task, head in self.heads.items()
        }
        
        return outputs