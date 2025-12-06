import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

# PyTorch Geometrics Imports (Compatible con v2.4+)
from torch_geometric.nn import (
    GATv2Conv, 
    GINEConv, 
    HeteroConv, 
    Linear, 
    global_add_pool, 
    global_mean_pool
)
from torch_geometric.data import HeteroData, Batch, Data

# =============================================================================
# 1. DRUG ENCODER (Homogeneous Graph - Molecular Structure)
# =============================================================================

class DrugGNNEncoder(nn.Module):
    """
    Codifica grafos moleculares (Fármacos).
    Usa GINEConv si hay características de enlace, o GATv2Conv si nos basamos en atención.
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int, 
        out_dim: int, 
        num_layers: int = 3,
        dropout: float = 0.1,
        gnn_type: str = "gine"  # Opciones: 'gine', 'gat'
    ):
        super().__init__()
        self.gnn_type = gnn_type
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Proyección inicial para alinear dimensiones
        self.input_proj = Linear(in_channels, hidden_dim)

        for _ in range(num_layers):
            if gnn_type == "gine":
                # GINE: Graph Isomorphism Network con Edge Features
                # Requiere un MLP interno
                mlp = nn.Sequential(
                    Linear(hidden_dim, hidden_dim * 2), 
                    nn.BatchNorm1d(hidden_dim * 2),
                    nn.ReLU(), 
                    Linear(hidden_dim * 2, hidden_dim)
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
            else:
                # GATv2: Atención dinámica
                self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=4, concat=False, dropout=dropout))
            
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.output_proj = Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        # x shape: [Num_Atoms, In_Features]
        x = self.input_proj(x)
        
        for i, conv in enumerate(self.convs):
            if self.gnn_type == "gine":
                if edge_attr is None:
                    # Fallback de seguridad: crear dummy edge attributes si faltan
                    edge_attr = torch.zeros((edge_index.size(1), x.size(1)), device=x.device)
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            
            x = self.bns[i](x)
            x = F.elu(x)
            x = self.dropout(x)

        # Global Pooling: Sumamos los átomos para obtener el vector molecular
        # global_add_pool es preferible en química (propiedad extensiva)
        x_pool = global_add_pool(x, batch)
        
        return self.output_proj(x_pool)

# =============================================================================
# 2. GENOTYPE ENCODER (Heterogeneous Graph - SNP -> Haplotype -> Gene)
# =============================================================================

class GenotypeHeteroEncoder(nn.Module):
    """
    Codifica la jerarquía genómica del paciente.
    Flujo de información: SNP (Variante) -> Haplotype (Alelo) -> Gene (Entidad Funcional).
    """
    def __init__(
        self, 
        hidden_dim: int, 
        out_dim: int, 
        metadata: Tuple[list, list], # (node_types, edge_types)
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        node_types, edge_types = metadata
        
        # 1. Proyección Inicial (Alinear SNP, Haplo y Gene al mismo espacio)
        # Usamos Linear(-1, ...) para inferencia "lazy" de dimensiones de entrada
        self.lin_dict = nn.ModuleDict({
            nt: Linear(-1, hidden_dim) for nt in node_types
        })

        # 2. Convoluciones Heterogéneas
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # Creamos un dict de convoluciones para cada tipo de arista
            conv_dict = {}
            for edge_type in edge_types:
                # edge_type es una tupla: (src_node, relation, dst_node)
                # Usamos GATv2 para aprender qué variantes son más importantes
                conv_dict[edge_type] = GATv2Conv(
                    hidden_dim, hidden_dim, heads=2, concat=False, add_self_loops=False
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        # 3. Salida
        self.out_proj = Linear(hidden_dim, out_dim)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        """
        batch_dict: Diccionario que contiene el vector 'batch' para cada tipo de nodo.
                    Ej: batch_dict['gene'] = [0, 0, 1, 1, 1...]
        """
        
        # A. Proyección
        x_dict = {key: self.lin_dict[key](x) for key, x in x_dict.items()}

        # B. Message Passing
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Activación + Dropout por tipo de nodo
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
            x_dict = {key: self.dropout(x) for key, x in x_dict.items()}

        # C. Readout (Pooling)
        # Nos interesa colapsar la información en el nivel del PACIENTE.
        # Asumimos que el nodo 'gene' ha agregado la info de sus haplotipos y SNPs.
        if 'gene' not in x_dict:
            raise ValueError("El grafo debe contener nodos tipo 'gene' para el readout final.")

        gene_features = x_dict['gene']
        gene_batch = batch_dict['gene'] # Vector que asigna cada gen a un paciente (batch idx)

        # Sumamos todos los genes de un paciente para obtener su representación final
        patient_vector = global_add_pool(gene_features, gene_batch)
        
        return self.out_proj(patient_vector)

# =============================================================================
# 3. UNIFIED ARCHITECTURE (DeepFM + Dual Graph)
# =============================================================================

class PharmagenModel(nn.Module):
    def __init__(
        self,
        # Feature Configs
        n_categorical_features: Dict[str, int], 
        target_dims: Dict[str, int],
        
        # Graph Metadata needed for initialization
        genotype_metadata: Tuple[list, list],
        drug_in_features: int,
        
        # Hyperparameters
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        n_layers: int = 3,
        
        # DeepFM Config
        fm_hidden_layers: int = 1
    ):
        super().__init__()
        self.categorical_names = list(n_categorical_features.keys())
        
        # --- 1. EMBEDDINGS ---
        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num, embedding_dim) 
            for feat, num in n_categorical_features.items()
        })

        # --- 2. GRAPH ENCODERS ---
        self.drug_encoder = DrugGNNEncoder(
            in_channels=drug_in_features,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            gnn_type="gine"
        )

        self.genotype_encoder = GenotypeHeteroEncoder(
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            metadata=genotype_metadata
        )

        # --- 3. INTERACTION LAYER (Transformer) ---
        self.emb_dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- 4. PREDICTION HEADS (Deep + FM) ---
        
        # Input dim for Deep MLP = (Num_Cats + Drug + Genotype) * Emb_Dim
        total_tokens = len(n_categorical_features) + 2 
        deep_input_dim = total_tokens * embedding_dim

        # Deep Component
        self.deep_mlp = nn.Sequential(
            Linear(deep_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )

        # FM Component (Optional extra MLP for the FM part)
        self.fm_mlp = None
        if fm_hidden_layers > 0:
            self.fm_mlp = nn.Sequential(
                Linear(embedding_dim, embedding_dim),
                nn.ReLU(),
                Linear(embedding_dim, embedding_dim)
            )

        # Output Heads
        final_dim = (hidden_dim // 2) + embedding_dim # Deep + FM
        self.heads = nn.ModuleDict({
            target: Linear(final_dim, dim) 
            for target, dim in target_dims.items()
        })
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(
        self, 
        x_cat: Dict[str, torch.Tensor], 
        drug_data: Batch, 
        genotype_data: Batch # Realmente es un Batch de HeteroData
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Obtener Embeddings Categóricos [Batch, N_Cats, Emb_Dim]
        cat_embs = [self.cat_embeddings[f](x_cat[f]) for f in self.categorical_names]
        
        # 2. Obtener Embedding de Fármaco [Batch, Emb_Dim]
        drug_emb = self.drug_encoder(
            x=drug_data.x, 
            edge_index=drug_data.edge_index, 
            batch=drug_data.batch,
            edge_attr=getattr(drug_data, 'edge_attr', None)
        )

        # 3. Obtener Embedding de Genotipo [Batch, Emb_Dim]
        # Extraemos el batch dict necesario para el pooling por tipos
        batch_dict = {
            key: genotype_data[key].batch 
            for key in genotype_data.node_types
        }
        
        genotype_emb = self.genotype_encoder(
            x_dict=genotype_data.x_dict,
            edge_index_dict=genotype_data.edge_index_dict,
            batch_dict=batch_dict
        )

        # 4. Stackear (Tokens) [Batch, Total_Tokens, Emb_Dim]
        # Añadimos unsqueeze(1) para que tengan dimensión de secuencia
        all_embs = cat_embs + [drug_emb.unsqueeze(1), genotype_emb.unsqueeze(1)]
        emb_stack = torch.cat(all_embs, dim=1)
        emb_stack = self.emb_dropout(emb_stack)

        # 5. Transformer Interaction
        trans_out = self.transformer(emb_stack)

        # 6. Deep Path
        deep_out = self.deep_mlp(trans_out.flatten(1))

        # 7. FM Path (Second Order Interactions)
        # FM standard formula: 0.5 * ( sum(x)^2 - sum(x^2) )
        sum_sq = torch.sum(emb_stack, dim=1).pow(2)
        sq_sum = torch.sum(emb_stack.pow(2), dim=1)
        fm_out = 0.5 * (sum_sq - sq_sum)
        
        if self.fm_mlp:
            fm_out = self.fm_mlp(fm_out)

        # 8. Combine & Predict
        combined = torch.cat([deep_out, fm_out], dim=-1)
        
        return {t: head(combined) for t, head in self.heads.items()}

# =============================================================================
# 4. TESTING BLOCK (Sanity Check)
# =============================================================================

if __name__ == "__main__":
    print("Inicializando Pharmagen Model Test...")
    
    # 1. Configuración Dummy
    n_features = {"age_group": 5, "hospital_id": 10}
    target_dims = {"solubility": 1, "toxicity": 2}
    
    # 2. Crear Datos Dummy
    batch_size = 4
    
    # A. Cats
    x_cat = {
        "age_group": torch.randint(0, 5, (batch_size,)),
        "hospital_id": torch.randint(0, 10, (batch_size,))
    }
    
    # B. Drug Graph (Homogeneous)
    # 4 grafos, cada uno con ~10 átomos
    from torch_geometric.data import Data
    drug_list = [
        Data(
            x=torch.randn(10, 9), # 9 features atómicos
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.randn(20, 3) # 3 features de enlace
        ) for _ in range(batch_size)
    ]
    drug_batch = Batch.from_data_list(drug_list)
    
    # C. Genotype Graph (Heterogeneous)
    # Jerarquía: SNP -> Haplotype -> Gene
    metadata = (
        ['snp', 'haplotype', 'gene'],
        [('snp', 'part_of', 'haplotype'), ('haplotype', 'defines', 'gene')]
    )
    
    geno_list = []
    for _ in range(batch_size):
        hdata = HeteroData()
        hdata['snp'].x = torch.randn(20, 3) # 20 SNPs, 3 dims (Ref, Het, Alt)
        hdata['haplotype'].x = torch.randn(5, 8) # 5 Haplos
        hdata['gene'].x = torch.randn(2, 16) # 2 Genes
        
        # Aristas dummy
        hdata['snp', 'part_of', 'haplotype'].edge_index = torch.randint(0, 5, (2, 15))
        hdata['haplotype', 'defines', 'gene'].edge_index = torch.randint(0, 2, (2, 5))
        geno_list.append(hdata)
        
    geno_batch = Batch.from_data_list(geno_list)
    
    # 3. Instanciar Modelo
    model = PharmagenModel(
        n_categorical_features=n_features,
        target_dims=target_dims,
        genotype_metadata=metadata,
        drug_in_features=9, # Coincide con drug_list.x
        embedding_dim=64,
        hidden_dim=64
    )
    
    # 4. Forward Pass
    outputs = model(x_cat, drug_batch, geno_batch)
    
    print("\n--- Resultados del Test ---")
    for k, v in outputs.items():
        print(f"Target: {k} | Shape: {v.shape}")
        assert v.shape[0] == batch_size
        
    print("\n¡El esqueleto del modelo funciona correctamente!")
    def __init__(
        self,
        # Configuración General
        n_categorical_features: Dict[str, int], 
        target_dims: Dict[str, int],
        
        # Config Genotipo (Hetero)
        genotype_metadata: tuple, # (data.node_types, data.edge_types)
        
        # Config Fármaco (Homo)
        drug_in_features: int,
        
        # Hyperparams
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        n_layers: int = 3
    ):
        super().__init__()
        
        # 1. Categorical Embeddings (Contexto Clínico)
        self.cat_embeddings = nn.ModuleDict({
            feat: nn.Embedding(num, embedding_dim) 
            for feat, num in n_categorical_features.items()
        })
        self.categorical_names = list(n_categorical_features.keys())

        # 2. Drug Encoder (Standard GNN)
        self.drug_encoder = DrugGNNEncoder( # La clase que definimos en la respuesta anterior
            in_channels=drug_in_features,
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            gnn_type="gine" # Recomendado para moléculas pequeñas
        )

        # 3. Genotype Encoder (NEW HETERO)
        self.genotype_encoder = GenotypeHeteroEncoder(
            hidden_dim=hidden_dim,
            out_dim=embedding_dim,
            metadata=genotype_metadata
        )

        # 4. Interaction Transformer
        self.emb_dropout = nn.Dropout(dropout_rate)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4, dim_feedforward=hidden_dim*4,
            dropout=dropout_rate, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 5. Prediction Head (Deep + FM logic)
        total_tokens = len(n_categorical_features) + 2 # Cats + Drug + Genotype
        deep_input_dim = total_tokens * embedding_dim
        
        self.deep_mlp = nn.Sequential(
            nn.Linear(deep_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Heads Multi-task
        self.heads = nn.ModuleDict({
            t: nn.Linear(hidden_dim // 2, d) for t, d in target_dims.items()
        })

    def forward(
        self, 
        x_cat: Dict[str, torch.Tensor], 
        drug_data: Any,      # Batch de grafos de fármacos
        genotype_data: Any   # Batch de grafos heterogéneos (pacientes)
    ) -> Dict[str, torch.Tensor]:
        
        # --- A. Procesar Inputs ---
        
        # 1. Categoricos -> [Batch, N_Cats, Emb_Dim]
        cat_embs = [self.cat_embeddings[k](x_cat[k]) for k in self.categorical_names]
        
        # 2. Drug Graph -> [Batch, Emb_Dim]
        drug_emb = self.drug_encoder(
            drug_data.x, drug_data.edge_index, drug_data.batch, 
            edge_attr=getattr(drug_data, 'edge_attr', None)
        )
        
        # 3. Genotype Hetero Graph -> [Batch, Emb_Dim]
        # Pasamos los diccionarios internos del objeto HeteroData
        genotype_emb = self.genotype_encoder(
            genotype_data.x_dict, 
            genotype_data.edge_index_dict,
            genotype_data.batch_dict if hasattr(genotype_data, 'batch_dict') else {k: v for k,v in genotype_data.__dict__.items() if 'batch' in k}
            # Nota: PyG maneja los batches heterogeneos guardando un vector 'batch' por nodo tipo.
            # ej: genotype_data['gene'].batch
        )

        # --- B. Interacción y Predicción ---
        
        # Stack: [Batch, Tokens, Dim]
        # Tokens = [Cat1, Cat2..., Drug, Genotype]
        all_embs = cat_embs + [drug_emb.unsqueeze(1), genotype_emb.unsqueeze(1)]
        stack = torch.cat(all_embs, dim=1)
        stack = self.emb_dropout(stack)
        
        # Transformer
        trans_out = self.transformer(stack)
        
        # Deep MLP
        deep_out = self.deep_mlp(trans_out.flatten(1))
        
        # Outputs
        return {k: head(deep_out) for k, head in self.heads.items()}