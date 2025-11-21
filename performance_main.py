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

import os
import sys
import logging
import multiprocessing
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import networkx as nx
import numpy as np
import torch
import joblib
import gensim
from gensim.models import Word2Vec
from tqdm import tqdm
from rapidfuzz import process, fuzz

# Imports condicionales para el método BioBERT
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None

# Imports condicionales para Graph/PecanPy
try:
    from pecanpy import pecanpy
except ImportError:
    pecanpy = None

# Configurar Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# CONFIGURACIÓN GLOBAL
# ==============================================================================

# --- SELECCIÓN DEL MÉTODO ---
# Opciones: 'GRAPH' (Node2Vec + Gensim) o 'BIOBERT' (Pre-trained LM)
EMBEDDING_METHOD = 'GRAPH' 

# --- Rutas ---
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data' # Ajusta según tu estructura
MODEL_DIR = BASE_DIR / 'models'
ENCODER_DIR = BASE_DIR / 'encoders'

SNP_DATA_FILE = DATA_DIR / 'snp_summary.tsv'
DRUG_DATA_FILE = DATA_DIR / 'drug_gene_edges.tsv'
ENCODER_FILE = ENCODER_DIR / 'encoders_Phenotype_Effect_Outcome.pkl'

# Salidas Intermedias y Finales
KGE_FILE = ENCODER_DIR / 'pgx_embeddings.kv' # Salida intermedia del Grafo
OUTPUT_WEIGHTS_FILE = MODEL_DIR / 'pretrained_weights.pth' # Salida final para PyTorch

# --- Parámetros Graph/Word2Vec ---
ENTITY_COLUMNS = ['snp', 'gene', 'Alt_Allele', 'chr', 'clin_sig']
EMBEDDING_DIM = 128
WALK_LENGTH = 30
NUM_WALKS = 150
WINDOW_SIZE = 10
MIN_COUNT = 1
WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 8))

# --- Parámetros BioBERT ---
BIOBERT_MODEL_ID = "dmis-lab/biobert-v1.2" # o "dmis-lab/biobert-base-cased-v1.2"

# --- Parámetros de Mapeo ---
FUZZY_THRESHOLD = 90
LAYERS_TO_MAP = ['drug', 'genalle', 'gene', 'allele']


class PGxEmbeddingManager:
    """Clase orquestadora para la generación y mapeo de embeddings."""

    def __init__(self, method: str):
        self.method = method.upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._check_dependencies()
        
        # Crear directorios si no existen
        for d in [MODEL_DIR, ENCODER_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def _check_dependencies(self):
        if self.method == 'GRAPH' and pecanpy is None:
            logger.error("La librería 'pecanpy' es necesaria para el modo GRAPH. pip install pecanpy")
            sys.exit(1)
        if self.method == 'BIOBERT' and AutoTokenizer is None:
            logger.error("La librería 'transformers' es necesaria para el modo BIOBERT. pip install transformers")
            sys.exit(1)

    # ==========================================================================
    # LÓGICA 1: MÉTODO GRAPH (NODE2VEC)
    # ==========================================================================
    
    def _load_graph_data(self) -> nx.Graph:
        """Carga y construye el grafo a partir de los TSV."""
        logger.info("--- [GRAPH] Iniciando construcción del grafo ---")
        
        if not SNP_DATA_FILE.exists() or not DRUG_DATA_FILE.exists():
            raise FileNotFoundError(f"Faltan archivos de datos en {DATA_DIR}")

        # Cargar DataFrames
        logger.info("Cargando DataFrames...")
        snp_df = pd.read_csv(SNP_DATA_FILE, sep='\t', usecols=ENTITY_COLUMNS)
        snp_df.columns = snp_df.columns.str.strip()
        
        drug_df = pd.read_csv(DRUG_DATA_FILE, sep='\t')

        # Construcción Vectorizada del Grafo
        G = nx.Graph()
        
        # Pre-procesamiento de strings
        for col in ['gene', 'Alt_Allele', 'clin_sig']:
            snp_df[col] = snp_df[col].fillna('').astype(str).str.split(r'[;,]')

        # Función auxiliar para añadir bordes masivamente
        def add_exploded_edges(df, col_source, col_target, rel_name):
            exploded = df[[col_source, col_target]].explode(col_target).dropna()
            # Filtrar vacíos
            exploded = exploded[exploded[col_target].str.len() > 0]
            if not exploded.empty:
                G.add_edges_from(exploded.itertuples(index=False), relationship=rel_name)
                logger.info(f"Añadidos {len(exploded)} bordes: {rel_name}")

        add_exploded_edges(snp_df, 'snp', 'gene', 'snp_to_gene')
        add_exploded_edges(snp_df, 'snp', 'Alt_Allele', 'snp_to_allele')
        add_exploded_edges(snp_df, 'snp', 'clin_sig', 'snp_to_clin_sig')

        # Bordes Drug-Gene
        drug_edges = drug_df[['drug', 'gene']].dropna()
        G.add_edges_from(drug_edges.itertuples(index=False), relationship='drug_to_gene')
        logger.info(f"Añadidos {len(drug_edges)} bordes: drug_to_gene")

        # Bordes complejos Gene-ClinSig (Producto Cartesiano)
        logger.info("Procesando bordes Gene-ClinSig (puede tardar)...")
        gene_clinsig_df = snp_df[['gene', 'clin_sig']].dropna()
        edges_complex = []
        for row in tqdm(gene_clinsig_df.itertuples(index=False), total=len(gene_clinsig_df)):
            # Filter empty strings
            genes = [g for g in row.gene if g]
            sigs = [s for s in row.clin_sig if s]
            if genes and sigs:
                edges_complex.extend(product(genes, sigs))
        
        G.add_edges_from(edges_complex, relationship='gene_to_clin_sig')
        logger.info(f"Nodos: {G.number_of_nodes()}, Bordes: {G.number_of_edges()}")
        
        return G

    def _train_node2vec(self, G: nx.Graph) -> gensim.models.KeyedVectors:
        """Entrena Node2Vec usando PecanPy y Gensim."""
        logger.info("--- [GRAPH] Entrenando Node2Vec ---")
        
        # Mapeo String -> Int para PecanPy (C++)
        nodes = list(G.nodes())
        node_map = {node: i for i, node in enumerate(nodes)}
        reverse_map = {i: node for node, i in node_map.items()}
        G_int = nx.relabel_nodes(G, node_map)

        temp_edg = "temp_graph.edg"
        try:
            nx.write_edgelist(G_int, temp_edg, data=False, delimiter=' ', encoding='utf-8')
            
            # Inicializar PecanPy
            g_pecan = pecanpy.SparseOTF(p=1, q=1, workers=WORKERS, verbose=False)
            g_pecan.read_edg(temp_edg, weighted=False, directed=False, delimiter=' ')
            
            # Generar Caminos
            logger.info("Simulando caminatas aleatorias...")
            walks_int = g_pecan.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LENGTH)
            
            # Traducir Int -> String
            logger.info("Traduciendo caminatas...")
            walks_str = [[reverse_map[n] for n in walk] for walk in walks_int]
            
            # Word2Vec
            logger.info("Entrenando Word2Vec...")
            model = Word2Vec(
                walks_str, 
                vector_size=EMBEDDING_DIM, 
                window=WINDOW_SIZE, 
                min_count=MIN_COUNT, 
                workers=WORKERS, 
                sg=1
            )
            
            # Guardar KeyedVectors
            model.wv.save_word2vec_format(str(KGE_FILE))
            return model.wv

        finally:
            if os.path.exists(temp_edg):
                os.remove(temp_edg)

    # ==========================================================================
    # LÓGICA 2: MÉTODO BIOBERT
    # ==========================================================================

    def _get_biobert_embedding(self, text: str, tokenizer, model) -> torch.Tensor:
        """Genera embedding para un texto usando BioBERT."""
        # Tokenización
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Estrategia: Mean Pooling de la última capa oculta
        # outputs.last_hidden_state: [batch, seq_len, hidden_dim]
        # Promediamos sobre seq_len (dim 1) ignorando padding si quisiéramos ser estrictos,
        # pero para palabras cortas/entidades el mean simple suele funcionar bien.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.cpu()

    def _process_biobert_layers(self, encoders: Dict) -> Dict[str, torch.Tensor]:
        """Genera embeddings directamente desde el vocabulario de los encoders."""
        logger.info(f"--- [BIOBERT] Cargando modelo {BIOBERT_MODEL_ID} ---")
        tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL_ID)
        model = AutoModel.from_pretrained(BIOBERT_MODEL_ID).to(self.device)
        model.eval()

        weight_matrix_dict = {}

        for layer_name in LAYERS_TO_MAP:
            if layer_name not in encoders:
                continue

            vocab = encoders[layer_name].classes_
            vocab_size = len(vocab)
            # Dimensión de BioBERT base suele ser 768
            bert_dim = model.config.hidden_size 
            
            logger.info(f"Generando embeddings BioBERT para capa: {layer_name} ({vocab_size} items)")
            
            # Matriz contenedora
            matrix = torch.zeros((vocab_size, bert_dim), dtype=torch.float32)

            for i, entity in tqdm(enumerate(vocab), total=vocab_size, desc=layer_name):
                # Limpieza básica de entidad (ej: quitar prefijos si es necesario)
                clean_text = str(entity).replace("_", " ") 
                emb = self._get_biobert_embedding(clean_text, tokenizer, model)
                matrix[i] = emb
            
            weight_matrix_dict[layer_name] = matrix
            
        return weight_matrix_dict

    # ==========================================================================
    # LÓGICA COMÚN: MAPEO Y GUARDADO
    # ==========================================================================

    def _map_graph_embeddings(self, kge_wv, encoders: Dict) -> Dict[str, torch.Tensor]:
        """Mapea vectores pre-entrenados del grafo a los índices de los encoders."""
        logger.info("--- [MAPPING] Mapeando Grafos a Encoders ---")
        
        kge_keys = list(kge_wv.key_to_index.keys())
        embedding_dim = kge_wv.vector_size
        weight_matrix_dict = {}

        for layer_name in LAYERS_TO_MAP:
            if layer_name not in encoders:
                logger.warning(f"Encoder no encontrado para: {layer_name}")
                continue

            vocab = encoders[layer_name].classes_
            vocab_size = len(vocab)
            
            # Inicializar con ruido pequeño
            matrix = torch.randn(vocab_size, embedding_dim) * 0.01
            hits = 0

            for i, entity in tqdm(enumerate(vocab), total=vocab_size, desc=f"Mapping {layer_name}"):
                entity_str = str(entity)
                
                # 1. Match Exacto
                if entity_str in kge_wv:
                    matrix[i] = torch.tensor(kge_wv[entity_str].copy())
                    hits += 1
                    continue
                
                # 2. Match Difuso
                match = process.extractOne(
                    entity_str, kge_keys, scorer=fuzz.ratio, score_cutoff=FUZZY_THRESHOLD
                )
                if match:
                    best_match, _, _ = match
                    matrix[i] = torch.tensor(kge_wv[best_match].copy())
                    hits += 1
            
            logger.info(f"Capa {layer_name}: {hits}/{vocab_size} encontrados ({hits/vocab_size:.1%})")
            weight_matrix_dict[layer_name] = matrix
            
        return weight_matrix_dict

    def run(self):
        """Ejecuta el pipeline completo según la configuración."""
        
        # 1. Cargar Encoders (Necesarios para ambos métodos)
        if not ENCODER_FILE.exists():
            logger.error(f"No se encuentra el archivo de encoders: {ENCODER_FILE}")
            return

        logger.info(f"Cargando encoders: {ENCODER_FILE}")
        encoders = joblib.load(ENCODER_FILE)

        final_weights = {}

        if self.method == 'GRAPH':
            # A. Construir y Entrenar Grafo
            if KGE_FILE.exists():
                logger.info(f"Cargando embeddings de grafo existentes: {KGE_FILE}")
                kge_wv = gensim.models.KeyedVectors.load_word2vec_format(str(KGE_FILE))
            else:
                G = self._load_graph_data()
                if G.number_of_nodes() == 0:
                    logger.error("Grafo vacío.")
                    return
                kge_wv = self._train_node2vec(G)
            
            # B. Mapear
            final_weights = self._map_graph_embeddings(kge_wv, encoders)

        elif self.method == 'BIOBERT':
            # A. Generar Directamente
            final_weights = self._process_biobert_layers(encoders)

        else:
            logger.error(f"Método desconocido: {self.method}")
            return

        # 3. Guardar Resultado Final
        if final_weights:
            logger.info(f"Guardando pesos finales en: {OUTPUT_WEIGHTS_FILE}")
            torch.save(final_weights, OUTPUT_WEIGHTS_FILE)
            
            # Verificación rápida de dimensiones
            for k, v in final_weights.items():
                logger.info(f"Tensor guardado -> {k}: {v.shape}")
        else:
            logger.warning("No se generaron pesos. Revisa los logs.")


def main():
    manager = PGxEmbeddingManager(method=EMBEDDING_METHOD)
    manager.run()

if __name__ == "__main__":
    main()