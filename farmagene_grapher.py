import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, global_mean_pool
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import numpy as np
import pandas as pd
from pyfaidx import Fasta
import re

# Diccionario de conversión: Nombre Común -> Nombre NCBI RefSeq (GRCh38)
CHROM_MAPPING = {
    'chr1': 'NC_000001.11', '1': 'NC_000001.11',
    'chr2': 'NC_000002.12', '2': 'NC_000002.12',
    'chr3': 'NC_000003.12', '3': 'NC_000003.12',
    'chr4': 'NC_000004.12', '4': 'NC_000004.12',
    'chr5': 'NC_000005.10', '5': 'NC_000005.10',
    'chr6': 'NC_000006.12', '6': 'NC_000006.12',
    'chr7': 'NC_000007.14', '7': 'NC_000007.14',
    'chr8': 'NC_000008.11', '8': 'NC_000008.11',
    'chr9': 'NC_000009.12', '9': 'NC_000009.12',
    'chr10': 'NC_000010.11', '10': 'NC_000010.11',
    'chr11': 'NC_000011.10', '11': 'NC_000011.10',
    'chr12': 'NC_000012.12', '12': 'NC_000012.12',
    'chr13': 'NC_000013.11', '13': 'NC_000013.11',
    'chr14': 'NC_000014.9',  '14': 'NC_000014.9',
    'chr15': 'NC_000015.10', '15': 'NC_000015.10',
    'chr16': 'NC_000016.10', '16': 'NC_000016.10',
    'chr17': 'NC_000017.11', '17': 'NC_000017.11',
    'chr18': 'NC_000018.10', '18': 'NC_000018.10',
    'chr19': 'NC_000019.10', '19': 'NC_000019.10',
    'chr20': 'NC_000020.11', '20': 'NC_000020.11',
    'chr21': 'NC_000021.9',  '21': 'NC_000021.9',
    'chr22': 'NC_000022.11', '22': 'NC_000022.11', # <--- ESTE ES EL QUE NECESITAS
    'chrX': 'NC_000023.11',  'X': 'NC_000023.11',
    'chrY': 'NC_000024.10',  'Y': 'NC_000024.10',
    'chrM': 'NC_012920.1',   'M': 'NC_012920.1', 'MT': 'NC_012920.1'
}

# ==========================================
# 1. MOTOR DE EMBEDDINGS (DNABERT-2)
# ==========================================
class GenomicEmbedder:
    def __init__(self, model_size="500m"):
        # Selección de modelo según tu potencia
        # Opciones: "500m" (rápido/equilibrado) o "2.5b" (máxima precisión)
        if model_size == "2.5b":
            self.model_name = "InstaDeepAI/nucleotide-transformer-2.5b-multi-species"
        else:
            self.model_name = "InstaDeepAI/nucleotide-transformer-500m-human-ref"

        self.device = torch.device("cuda") # Tu RTX 4070 Ti Super
        
        print(f"--- Cargando {self.model_name} en GPU con FP16 ---")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        
        # CLAVE: torch.float16 reduce la VRAM a la mitad. Tu tarjeta lo maneja nativamente.
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            torch_dtype=torch.float16 
        )
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, sequences, batch_size=64):
        # Con 16GB VRAM puedes subir el batch_size. Prueba 64 o 128 con el modelo 500m.
        # Si usas el 2.5B, baja el batch_size a 16 o 32.
        
        all_embeddings = []
        
        print(f"Procesando {len(sequences)} secuencias...")
        
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            
            inputs = self.tokenizer(
                batch_seqs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1000 
            ).to(self.device)
            
            # Usamos autocast para asegurar eficiencia máxima en los Tensor Cores
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            
            # Lógica específica de Nucleotide Transformer para embeddings
            # Usamos la última capa oculta
            last_hidden_state = outputs.hidden_states[-1] 
            
            # Mean pooling con máscara de atención (para ignorar padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            
            # Convertimos a float32 solo para la operación de suma/división final para evitar overflow
            masked_hidden = last_hidden_state * attention_mask
            sum_embeddings = torch.sum(masked_hidden, dim=1)
            sum_mask = torch.sum(attention_mask, dim=1)
            
            batch_emb = sum_embeddings / sum_mask
            
            # Pasamos a CPU para liberar VRAM inmediatamente
            all_embeddings.append(batch_emb.cpu().float())
            
        return torch.cat(all_embeddings, dim=0)

# ==========================================
# 2. CONSTRUCTOR DEL GRAFO FARMACOGENÓMICO
# ==========================================
class PharmacogeneGraphBuilder:
    def __init__(self, embedder, context_window=10):
        self.embedder = embedder
        self.k = context_window  # Cuantas bases mirar a izq/der del nodo

    def _get_context_seq(self, full_seq, center_idx, alt_base=None):
        """
        Extrae la subsecuencia alrededor de una posición.
        Si hay alt_base, simula la variante.
        """
        start = max(0, center_idx - self.k)
        end = min(len(full_seq), center_idx + self.k + 1)
        
        prefix = full_seq[start:center_idx]
        suffix = full_seq[center_idx+1:end]
        
        # El nucleótido central es la base original o la variante
        center = full_seq[center_idx] if alt_base is None else alt_base
        
        return prefix + center + suffix

    def build(self, ref_seq, variants, annotations):
        """
        ref_seq: String ADN
        variants: Lista [{'pos': int, 'alt': str, ...}]
        annotations: Lista [0, 1, ...] (0=Intrón, 1=Exón)
        """
        # --- VALIDACIÓN DE SEGURIDAD ---
        if len(ref_seq) != len(annotations):
            raise ValueError(
                f"Desajuste de longitud: La secuencia tiene {len(ref_seq)} bases "
                f"pero las anotaciones cubren {len(annotations)} posiciones."
            )
        # -------------------------------
        num_ref_nodes = len(ref_seq)
        
        # --- Listas para construir el grafo ---
        node_contexts = []    # Strings de ADN para pasar al Transformer
        node_regions = []     # 0=Intrón, 1=Exón
        edge_sources = []
        edge_targets = []
        edge_types = []       # 0=Ref, 1=Var
        
        # 1. Nodos de Referencia (Backbone)
        for i in range(num_ref_nodes):
            # Extraer contexto para el nodo i
            seq_context = self._get_context_seq(ref_seq, i)
            node_contexts.append(seq_context)
            node_regions.append(annotations[i])
            
            # Arista lineal hacia el siguiente nodo
            if i < num_ref_nodes - 1:
                edge_sources.append(i)
                edge_targets.append(i + 1)
                edge_types.append(0) # Tipo 0: Backbone
                
                # Bidireccional (opcional, recomendado para GNNs)
                edge_sources.append(i + 1)
                edge_targets.append(i)
                edge_types.append(0)

        # 2. Nodos de Variantes (Burbujas)
        current_max_idx = num_ref_nodes - 1
        
        for var in variants:
            pos = var['pos']
            alt = var['alt']
            
            # Crear contexto "mutado"
            var_context = self._get_context_seq(ref_seq, pos, alt_base=alt)
            node_contexts.append(var_context)
            
            # Asumimos que la variante hereda la región (exón/intrón)
            node_regions.append(annotations[pos])
            
            current_max_idx += 1
            var_node_idx = current_max_idx
            
            # Conectar la burbuja: (pos-1) -> VAR -> (pos+1)
            if pos > 0:
                edge_sources.append(pos - 1)
                edge_targets.append(var_node_idx)
                edge_types.append(1) # Tipo 1: Camino Variante
            
            if pos < num_ref_nodes - 1:
                edge_sources.append(var_node_idx)
                edge_targets.append(pos + 1)
                edge_types.append(1)

        # --- CONVERSIÓN A TENSORES ---
        print(f"Generando embeddings para {len(node_contexts)} nodos...")
        # AQUI OCURRE LA MAGIA: Convertimos texto a vectores numéricos (768 dims)
        x_embeddings = self.embedder.get_embeddings(node_contexts)
        
        # Concatenar la info de región (Exón/Intrón) al embedding
        # x_embeddings es [N, 768], region es [N, 1]
        x_region = torch.tensor(node_regions, dtype=torch.float).unsqueeze(1)
        x = torch.cat([x_embeddings, x_region], dim=1) # Dim total: 769
        
        edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
        edge_attr = torch.tensor(edge_types, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

# ==========================================
# 3. MODELO GNN (El Clasificador)
# ==========================================
class PharmacoGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # GATv2Conv es potente para mecanismos de atención en grafos
        # edge_dim=1 permite usar el edge_attr (tipo de arista)
        self.conv1 = GATv2Conv(input_dim, hidden_dim, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=1)
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # El edge_attr necesita ser [Num_Edges, 1] o [Num_Edges, Dim]
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(1)

        # Capa 1
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Capa 2
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        
        # Global Pooling: Resumir todo el grafo (el gen entero) en un vector
        # Usamos mean_pool, pero para farmacogenética max_pool o attention_pool también sirven
        x = global_mean_pool(x, data.batch if hasattr(data, 'batch') else None)
        
        # Clasificación final
        out = self.classifier(x)
        return out

class GenomeDataManager:
    def __init__(self, fasta_path, gtf_path):
        print(f"--- Indexando FASTA: {fasta_path} ---")
        self.genome = Fasta(fasta_path)
        
        print(f"--- Cargando GTF: {gtf_path} ---")
        cols = ['seqname', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attribute']
        
        # Cargamos el GTF
        self.gtf = pd.read_csv(
            gtf_path, 
            sep='\t', 
            comment='#', 
            names=cols, 
            low_memory=False
        )
        
        # OPTIMIZACIÓN: Nos quedamos solo con exones para aligerar
        self.gtf = self.gtf[self.gtf['feature'] == 'exon'].copy()

        print("--- Procesando atributos del GTF (Extrayendo nombres de genes) ---")
        # AQUI ESTÁ LA CLAVE: Creamos la columna 'gene_name' que te falta
        self.gtf['gene_name'] = self.gtf['attribute'].apply(self._extract_gene_name)
        
        # Depuración: Verificamos que se creó bien
        print(f"Muestra de genes extraídos: {self.gtf['gene_name'].dropna().unique()[:5]}")

    def _extract_gene_name(self, attribute_string):
        """
        Intenta extraer el nombre del gen soportando formato GENCODE y NCBI.
        """
        # Intento 1: Formato estándar GENCODE (gene_name "CYP2D6";)
        match = re.search(r'gene_name "([^"]+)"', attribute_string)
        if match:
            return match.group(1)
            
        # Intento 2: Formato NCBI RefSeq (gene "CYP2D6"; o gene_id "CYP2D6";)
        match = re.search(r'gene "([^"]+)"', attribute_string)
        if match:
            return match.group(1)
            
        # Intento 3: A veces en NCBI el nombre está en 'gene_id'
        match = re.search(r'gene_id "([^"]+)"', attribute_string)
        if match:
            return match.group(1)

        return None

    def extract_gene_data(self, chromosome_common, start_pos, end_pos, target_gene=None):
        # 1. Obtener clave FASTA usando el mapeo
        fasta_key = CHROM_MAPPING.get(chromosome_common, chromosome_common)
        
        try:
            # Extraer secuencia (1-based a 0-based)
            sequence_obj = self.genome[fasta_key][start_pos-1 : end_pos]
            ref_seq = sequence_obj.seq.upper()
        except KeyError:
            valid = list(self.genome.keys())[:3]
            raise ValueError(f"FASTA Key '{fasta_key}' no encontrada. Disponibles: {valid}")

        # 2. Filtrar GTF (usando el nombre común del cromosoma)
        subset = self.gtf[
            (self.gtf['seqname'] == chromosome_common) &
            (self.gtf['start'] >= start_pos) & 
            (self.gtf['end'] <= end_pos)
        ]
        
        # 3. Filtrar por nombre de gen (AQUÍ DABA EL ERROR ANTES)
        if target_gene:
            # Verificamos si la columna tiene datos
            if 'gene_name' not in subset.columns:
                 raise KeyError("La columna 'gene_name' no se creó en el __init__.")
            
            subset = subset[subset['gene_name'] == target_gene]
            
        if subset.empty:
            print(f"[AVISO] No se encontraron exones para {target_gene} en {chromosome_common}:{start_pos}-{end_pos}")

        # 4. Construir Máscara
        seq_len = len(ref_seq)
        anno_mask = [0] * seq_len
        
        for _, row in subset.iterrows():
            loc_start = max(0, row['start'] - start_pos)
            loc_end = min(seq_len, row['end'] - start_pos + 1)
            
            for i in range(loc_start, loc_end):
                anno_mask[i] = 1 # Exón

        return ref_seq, anno_mask
# ==========================================
# 4. EJECUCIÓN (MAIN)
# ==========================================

def create_mask_from_coords(total_length, exon_ranges):
    """
    total_length: int (largo de ref_seq)
    exon_ranges: lista de tuplas [(start, end), (start, end)]
    Retorna: lista de 0s y 1s
    """
    # Iniciamos todo como Intrón (0)
    mask = [0] * total_length
    
    for start, end in exon_ranges:
        # Asegurar límites
        s = max(0, start)
        e = min(total_length, end)
        for i in range(s, e):
            mask[i] = 1 # Marcar zona de exón
            
    return mask

"""
if __name__ == "__main__":
    # --- 1. CONFIGURACIÓN ---
    PATH_FASTA = "data/ref_genome/GCF_000001405.40_GRCh38.p14_genomic.fa" 
    PATH_GTF = "data/ref_genome/genomic.gtf"
    
    # --- 2. CARGA DE DATOS REALES ---
    # Instanciamos el gestor (Hazlo solo una vez, cargar el GTF tarda unos segundos)
    data_manager = GenomeDataManager(PATH_FASTA, PATH_GTF)
    
    # DEFINIR REGIÓN DE INTERÉS (ROI)
    # Ejemplo: Gen CYP2D6 (Cromosoma 22)
    # Coordenadas aprox (GRCh38): chr22:42,126,000 - 42,131,000
    target_chrom = 'chr22'  # Ojo: revisa si tu fasta usa '22' o 'chr22'
    target_start = 42126000
    target_end = 42131000
    gene_name = "CYP2D6"

    print(f"\n--- Extrayendo {gene_name} de datos reales ---")
    real_seq, real_mask = data_manager.extract_gene_data(
        target_chrom, 
        target_start, 
        target_end, 
        target_gene=gene_name
    )
    
    # Validar que no haya pasado lo de antes
    if len(real_seq) != len(real_mask):
        raise ValueError("Error crítico en la generación de máscara.")

    # --- 3. PROCESAMIENTO CON EL MODELO ---
    # Inicializar tu embedder y builder como antes
    embedder = GenomicEmbedder(model_size="500m") # Tu GPU vuela aquí
    builder = PharmacogeneGraphBuilder(embedder, context_window=6)
    
    # Variante simulada en coordenadas RELATIVAS
    # Si la variante real está en 42,126,500 y cortamos desde 42,126,000
    # La pos relativa es 500.
    variants_patient = [
        {'pos': 500, 'alt': 'A', 'effect': 'missense'} 
    ]
    
    print("\n--- Construyendo Grafo Pangenómico ---")
    graph = builder.build(real_seq, variants_patient, real_mask)
    
    print(f"Grafo generado con {graph.num_nodes} nodos. Listo para GNN.")
"""