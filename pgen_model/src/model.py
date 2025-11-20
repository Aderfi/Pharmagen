import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer

logger = logging.getLogger(__name__)


class DeepFM_PGenModel(nn.Module):
    """
    Modelo DeepFM generalizado para predicción farmacogenética multitarea.
    """

    def __init__(
        self,
        n_features: Dict[str, int],
        target_dims: Dict[str, int],
        embedding_dim: int,
        hidden_dim: int,
        dropout_rate: float,
        n_layers: int,
        attention_dim_feedforward: Optional[int] = None,
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
        self.target_dims = target_dims

        # 1. Embeddings Dinámicos
        self.embeddings = nn.ModuleDict()
        for feat, num_classes in n_features.items():
            self.embeddings[feat] = nn.Embedding(num_classes, embedding_dim)

        self.emb_dropout = nn.Dropout(embedding_dropout)

        # 2. Deep Branch (Transformer + MLP)
        # Transformer espera: (Batch, Seq_Len, Emb_Dim) -> Aquí Seq_Len = n_fields
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

        # MLP Layers
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

        # 3. FM Branch (Interacciones de segundo orden)
        fm_input_dim = (self.n_fields * (self.n_fields - 1)) // 2
        self.fm_dropout = nn.Dropout(fm_dropout)

        if fm_hidden_layers > 0:
            self.fm_mlp = self._build_mlp(
                input_dim=fm_input_dim,
                hidden_dim=fm_hidden_dim,
                n_layers=fm_hidden_layers,
                dropout=fm_dropout,
                bn=use_batch_norm,
                ln=False,  # LayerNorm no usual en FM branch clásica
                act_fn=activation_function,
            )
            self.fm_out_dim = fm_hidden_dim
        else:
            self.fm_mlp = None
            self.fm_out_dim = fm_input_dim

        # 4. Output Heads & Uncertainty Weighting
        combined_dim = hidden_dim + self.fm_out_dim
        self.heads = nn.ModuleDict()
        self.log_sigmas = nn.ParameterDict()  # Learnable weights

        for target, dim in target_dims.items():
            self.heads[target] = nn.Linear(combined_dim, dim)
            self.log_sigmas[target] = nn.Parameter(torch.zeros(1))

    def _build_mlp(self, input_dim, hidden_dim, n_layers, dropout, bn, ln, act_fn):
        layers = []
        in_d = input_dim

        activation_map = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "mish": nn.Mish,
        }
        Act = activation_map.get(act_fn.lower(), nn.GELU)

        for _ in range(n_layers):
            layers.append(nn.Linear(in_d, hidden_dim))
            if bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if ln:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(Act())
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
        self, inputs: Dict[str, torch.Tensor], **kwargs
    ) -> Dict[str, torch.Tensor]:
        # Combinar kwargs si se pasan argumentos sueltos
        if kwargs:
            inputs.update(kwargs)

        # 1. Obtener Embeddings
        # Lista de tensores [Batch, Emb_Dim]
        emb_list = []
        for feat_name in self.feature_names:
            if feat_name in inputs:
                emb = self.embeddings[feat_name](inputs[feat_name])
                emb_list.append(emb)
            else:
                raise ValueError(f"Feature '{feat_name}' missing in inputs")

        # Stack para Transformer: [Batch, N_Fields, Emb_Dim]
        emb_stack = torch.stack(emb_list, dim=1)
        emb_stack = self.emb_dropout(emb_stack)

        # 2. Deep Path (Transformer -> Flatten -> MLP)
        trans_out = self.transformer(emb_stack)
        deep_in = trans_out.flatten(start_dim=1)
        deep_out = self.deep_mlp(deep_in)

        # 3. FM Path (Producto punto de pares de embeddings crudos)
        # fm_interactions shape: [Batch, N_Pairs, 1] -> squeeze -> [Batch, N_Pairs]
        fm_products = []
        for i in range(self.n_fields):
            for j in range(i + 1, self.n_fields):
                product = torch.sum(emb_list[i] * emb_list[j], dim=-1, keepdim=False)
                fm_products.append(product)

        fm_out = torch.stack(fm_products, dim=1)
        fm_out = self.fm_dropout(fm_out)

        if self.fm_mlp:
            fm_out = self.fm_mlp(fm_out)

        # 4. Concatenación y Salida
        combined = torch.cat([deep_out, fm_out], dim=-1)

        outputs = {name: head(combined) for name, head in self.heads.items()}
        return outputs

    def get_weighted_loss(
        self,
        losses: Dict[str, torch.Tensor],
        priorities: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        """
        Calcula la pérdida total usando ponderación de incertidumbre (Kendall & Gal).
        Loss = Loss_i * exp(-sigma_i) + sigma_i
        """
        total_loss = torch.tensor(0.0)
        for task, loss in losses.items():
            log_sigma = self.log_sigmas[task]

            # Prioridad clínica manual (multiplicador escalar)
            priority = priorities.get(task, 1.0) if priorities else 1.0

            # Cálculo numéricamente estable
            # precision = torch.exp(-log_sigma)
            # weighted_loss = precision * (loss * priority) + log_sigma

            # Factorización para eficiencia
            weighted_loss = (loss * priority) * torch.exp(-log_sigma) + log_sigma
            total_loss = total_loss + weighted_loss  # Acumulador tensor

        return total_loss

    def get_geometric_loss(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calcula la pérdida total usando la Estrategia de Pérdida Geométrica.
        Esta estrategia penaliza desproporcionadamente el mal rendimiento en cualquier tarea individual,
        forzando al modelo a mejorar en todas las tareas simultáneamente.

        Matemáticamente equivalente a: (Product(Loss_i)) ^ (1/N)
        Implementado en log-space para estabilidad numérica: exp( (1/N) * Sum(log(Loss_i)) )
        """
        if not losses:
            return torch.tensor(0.0)

        # Usamos el dispositivo del primer tensor de pérdida
        device = next(iter(losses.values())).device
        log_sum = torch.tensor(0.0, device=device)

        for loss in losses.values():
            # Añadimos un epsilon pequeño y clipeamos para evitar log(0) o inestabilidad
            safe_loss = torch.clamp(loss, min=1e-7)
            log_sum = log_sum + torch.log(safe_loss)

        # Media de los logaritmos
        mean_log_loss = log_sum / len(losses)

        # Exponencial para volver a la escala original (Media Geométrica)
        return torch.exp(mean_log_loss)

    def load_from_gensim_kv(
        self,
        kv_model: "KeyedVectors",
        feature_vocabs: Dict[
            str, Dict[str, int]
        ],  # Maps feature_name to its vocab dict
        freeze: bool = False,
    ):
        """
        Carga embeddings pre-entrenados desde un objeto KeyedVectors de Gensim.

        Este método mapea palabras de un único modelo KV a las capas
        de embedding correctas usando los diccionarios de vocabulario proporcionados.

        Args:
            kv_model: El modelo KeyedVectors de Gensim (ya cargado).
            feature_vocabs: Diccionario que mapea nombres de características a sus vocabularios.
                            Ejemplo: {'drug': {'drug_name': index}, 'gene': {'gene_name': index}}
            freeze: Si es True, congela los pesos de los embeddings.
        """
        try:
            from gensim.models import KeyedVectors
        except ImportError:
            logger.error("Gensim no está instalado. No se pueden cargar embeddings KV.")
            return

        logger.info("Iniciando carga de embeddings desde modelo Gensim KV...")

        for feature_name, embedding_layer in self.embeddings.items():
            if feature_name not in feature_vocabs:
                logger.warning(
                    f"No se proporcionó vocabulario para la característica '{feature_name}'. Se omiten los embeddings pre-entrenados para esta capa."
                )
                continue

            vocab = feature_vocabs[feature_name]
            weights = embedding_layer.weight.data
            found = 0
            not_found = 0

            logger.info(f"Procesando capa: '{feature_name}'...")

            for word, index in vocab.items():
                if index >= embedding_layer.num_embeddings:
                    logger.warning(
                        f"Índice {index} para '{word}' en '{feature_name}' está fuera de rango ({embedding_layer.num_embeddings}). Omitiendo."
                    )
                    continue

                try:
                    vector = kv_model[word]
                    vector_tensor = torch.tensor(
                        vector, dtype=weights.dtype, device=weights.device
                    )

                    if vector_tensor.shape[0] != weights.shape[1]:
                        logger.error(
                            f"Error de dimensión en '{feature_name}' para '{word}': "
                            f"Modelo espera {weights.shape[1]}, KV tiene {vector_tensor.shape[0]}. Omitiendo."
                        )
                        continue

                    weights[index, :] = vector_tensor
                    found += 1

                except KeyError:
                    not_found += 1
                except Exception as e:
                    logger.error(
                        f"Error procesando '{word}' en capa '{feature_name}': {e}"
                    )

            logger.info(
                f"Capa '{feature_name}': {found} vectores cargados, {not_found} no encontrados (se mantienen aleatorios)."
            )

            embedding_layer.weight.data.copy_(weights)

            if freeze:
                embedding_layer.weight.requires_grad = False
                logger.info(f"Capa '{feature_name}' congelada (no-trainable).")

        logger.info("Carga de embeddings desde Gensim KV completada.")
