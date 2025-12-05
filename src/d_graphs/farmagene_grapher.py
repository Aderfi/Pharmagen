import torch
from torch_geometric.data import Data
import torch.nn as nn

# Mapeo: A, C, G, T, N
NUC_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

class PharmacogeneGraphBuilder:
    def __init__(self, embedding_dim=16):
        # En lugar de one-hot fijo, preparamos para usar embeddings aprendibles
        # 5 nucleótidos + 1 padding/unknown
        self.embedding_dim = embedding_dim
        self.nuc_embedding = nn.Embedding(num_embeddings=6, embedding_dim=embedding_dim)

    def sequence_to_indices(self, sequence):
        return torch.tensor([NUC_MAP.get(n, 4) for n in sequence], dtype=torch.long)

    def build_graph(self, ref_seq, variants, annotations):
        """
        ref_seq: String secuencia referencia.
        variants: Lista de dicts.
        annotations: Lista/Array de 0 (intron) y 1 (exon).
        """
        num_ref_nodes = len(ref_seq)
        
        # --- 1. Características de Nodos (Features) ---
        # Usamos índices enteros para luego pasar por un Embedding Layer en el modelo
        node_indices_ref = self.sequence_to_indices(ref_seq)
        
        # Característica explicita: Región (0=Intrón, 1=Exón)
        # Añadimos dimensión [N, 1]
        region_type = torch.tensor(annotations, dtype=torch.float).unsqueeze(1)
        
        # Inicializamos listas para nodos extra (variantes)
        extra_node_indices = []
        extra_region_types = []
        
        # --- 2. Construcción de Aristas y sus Tipos ---
        source = []
        target = []
        edge_types = [] # 0 = Backbone, 1 = Variante/Salto
        
        # Backbone lineal (Referencia)
        for i in range(num_ref_nodes - 1):
            source.append(i)
            target.append(i + 1)
            edge_types.append(0) # Tipo 0: Enlace normal
            
            # Opcional: Bidireccional
            source.append(i + 1)
            target.append(i)
            edge_types.append(0)

        # --- 3. Procesar Variantes (Burbujas) ---
        current_max_idx = num_ref_nodes - 1
        variant_mask = [0] * num_ref_nodes # 0 = Ref, 1 = Var

        for var in variants:
            pos = var['pos']
            alt_base = var['alt']
            
            # Crear característica para el nodo variante
            var_idx = NUC_MAP.get(alt_base, 4)
            extra_node_indices.append(var_idx)
            
            # Heredar región (Exón/Intrón) de la posición original
            # Nota: .item() para obtener el valor escalar del tensor
            extra_region_types.append(region_type[pos].item())
            
            current_max_idx += 1
            var_node_id = current_max_idx
            
            # Conexiones de la variante (La "burbuja")
            # Conectamos (pos-1) -> VAR -> (pos+1)
            # Usamos edge_type=1 para indicar que es un camino alternativo
            
            if pos > 0:
                source.append(pos - 1)
                target.append(var_node_id)
                edge_types.append(1) 
                
            if pos < num_ref_nodes - 1:
                source.append(var_node_id)
                target.append(pos + 1)
                edge_types.append(1)
            
            variant_mask.append(1) # Marcamos este nuevo nodo como variante

        # --- 4. Ensamblaje Final ---
        
        # Concatenar índices de nucleótidos (Ref + Vars)
        full_node_indices = torch.cat([
            node_indices_ref, 
            torch.tensor(extra_node_indices, dtype=torch.long)
        ])
        
        # Concatenar tipos de región
        if extra_region_types:
            full_region_types = torch.cat([
                region_type,
                torch.tensor(extra_region_types).unsqueeze(1)
            ], dim=0)
        else:
            full_region_types = region_type

        # Crear tensores de aristas
        edge_index = torch.tensor([source, target], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long) # Para usar en GNNs que aceptan tipos de arista (ej. RGCN)

        data = Data(
            x_indices=full_node_indices, # Input para el Embedding Layer
            x_region=full_region_types,  # Feature manual
            edge_index=edge_index,
            edge_attr=edge_attr,
            variant_mask=torch.tensor(variant_mask, dtype=torch.bool)
        )
        
        return data

# --- Ejemplo de Uso ---
builder = PharmacogeneGraphBuilder()
ref_seq = "ATGC" * 5 
anno_mask = [0]*5 + [1]*10 + [0]*5 
sample_variants = [{'pos': 10, 'alt': 'T'}] # Variante en exón

graph = builder.build_graph(ref_seq, sample_variants, anno_mask)

print(f"Nodos Totales: {graph.num_nodes}")
print(f"Aristas Totales: {graph.num_edges}")
print(f"Tipos de Arista disponibles: {graph.edge_attr.unique()}")
print("Nota: 'x_indices' está listo para pasar por una capa nn.Embedding dentro de tu modelo.")