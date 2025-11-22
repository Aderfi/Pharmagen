# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import sys
import logging
import multiprocessing
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

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
    from transformers import AutoTokenizer, AutoModel # type: ignore
except ImportError:
    AutoTokenizer = None
    AutoModel = None

# Imports condicionales para Graph/PecanPy
try:
    from pecanpy import pecanpy # type: ignore
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
EMBEDDING_METHOD = 'GRAPH' 

# --- Rutas ---
BASE_DIR = Path('.')
DATA_DIR = BASE_DIR / 'data'
MODEL_DIR = BASE_DIR / 'models'
ENCODER_DIR = BASE_DIR / 'encoders'

SNP_DATA_FILE = DATA_DIR / 'snp_summary.tsv'
DRUG_DATA_FILE = DATA_DIR / 'drug_gene_edges.tsv'
ENCODER_FILE = ENCODER_DIR / 'encoders_Phenotype_Effect_Outcome.pkl'

# Salidas
KGE_FILE = ENCODER_DIR / 'pgx_embeddings.kv'
OUTPUT_WEIGHTS_FILE = MODEL_DIR / 'pretrained_weights.pth'

# --- Parámetros Graph/Word2Vec ---
ENTITY_COLUMNS = ['snp', 'gene', 'Alt_Allele', 'chr', 'clin_sig']
EMBEDDING_DIM = 128
WALK_LENGTH = 30
NUM_WALKS = 150
WINDOW_SIZE = 10
MIN_COUNT = 1
WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 8))

# --- Parámetros BioBERT ---
BIOBERT_MODEL_ID = "dmis-lab/biobert-v1.2"

# --- Parámetros de Mapeo ---
FUZZY_THRESHOLD = 90
LAYERS_TO_MAP = ['drug', 'genalle', 'gene', 'allele']


class PGxEmbeddingManager:
    """Clase orquestadora para la generación y mapeo de embeddings."""

    def __init__(self, method: str):
        self.method = method.upper()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._check_dependencies()
        
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

        logger.info("Cargando DataFrames...")
        snp_df = pd.read_csv(SNP_DATA_FILE, sep='\t', usecols=ENTITY_COLUMNS) # type: ignore
        snp_df.columns = snp_df.columns.str.strip() # type: ignore
        
        drug_df = pd.read_csv(DRUG_DATA_FILE, sep='\t') # type: ignore

        G = nx.Graph()
        
        # Pre-procesamiento
        for col in ['gene', 'Alt_Allele', 'clin_sig']:
            snp_df[col] = snp_df[col].fillna('').astype(str).str.split(r'[;,]')

        def add_exploded_edges(df, col_source, col_target, rel_name):
            exploded = df[[col_source, col_target]].explode(col_target).dropna()
            exploded = exploded[exploded[col_target].str.len() > 0]
            
            if not exploded.empty:
                # Usamos zip para crear tuplas explícitas (u, v)
                edges = list(zip(exploded[col_source], exploded[col_target]))
                G.add_edges_from(edges, relationship=rel_name)
                logger.info(f"Añadidos {len(edges)} bordes: {rel_name}")

        add_exploded_edges(snp_df, 'snp', 'gene', 'snp_to_gene')
        add_exploded_edges(snp_df, 'snp', 'Alt_Allele', 'snp_to_allele')
        add_exploded_edges(snp_df, 'snp', 'clin_sig', 'snp_to_clin_sig')

        # Bordes Drug-Gene
        drug_edges_df = drug_df[['drug', 'gene']].dropna()
        drug_edges = list(zip(drug_edges_df['drug'], drug_edges_df['gene']))
        G.add_edges_from(drug_edges, relationship='drug_to_gene')
        logger.info(f"Añadidos {len(drug_edges)} bordes: drug_to_gene")

        # Bordes complejos Gene-ClinSig
        logger.info("Procesando bordes Gene-ClinSig (puede tardar)...")
        gene_clinsig_df = snp_df[['gene', 'clin_sig']].dropna()
        edges_complex = []
        
        for genes_raw, sigs_raw in tqdm(zip(gene_clinsig_df['gene'], gene_clinsig_df['clin_sig']), total=len(gene_clinsig_df)):
            genes = [g for g in genes_raw if g]  
            sigs = [s for s in sigs_raw if s]
            if genes and sigs:
                edges_complex.extend(product(genes, sigs))
        
        G.add_edges_from(edges_complex, relationship='gene_to_clin_sig')
        logger.info(f"Nodos: {G.number_of_nodes()}, Bordes: {G.number_of_edges()}")
        
        return G

    def _train_node2vec(self, G: nx.Graph) -> gensim.models.KeyedVectors:
        """Entrena Node2Vec usando PecanPy y Gensim."""
        logger.info("--- [GRAPH] Entrenando Node2Vec ---")
        
        # --- FIX: Asegurar a Pylance que pecanpy existe ---
        assert pecanpy is not None, "PecanPy no está instalado"

        nodes = list(G.nodes())
        # Mapa de String -> Int
        node_map = {node: i for i, node in enumerate(nodes)}
        # Mapa de Int -> String
        reverse_map: Dict[int, str] = {i: node for node, i in node_map.items()}
        
        G_int = nx.relabel_nodes(G, node_map)

        temp_edg = "temp_graph.edg"
        try:
            nx.write_edgelist(G_int, temp_edg, data=False, delimiter=' ', encoding='utf-8')
            
            g_pecan = pecanpy.SparseOTF(p=1, q=1, workers=WORKERS, verbose=False)
            g_pecan.read_edg(temp_edg, weighted=False, directed=False, delimiter=' ')
            
            logger.info("Simulando caminatas aleatorias...")
            # Simulate walks devuelve lista de lista de enteros (IDs de nodos)
            walks_int = g_pecan.simulate_walks(num_walks=NUM_WALKS, walk_length=WALK_LENGTH)

            logger.info("Traduciendo caminatas...")
            # --- FIX: List comprehension explícita para tipos ---
            walks_str = [[reverse_map[int(n)] for n in walk] for walk in walks_int]
            
            logger.info("Entrenando Word2Vec...")
            model = Word2Vec(
                walks_str, 
                vector_size=EMBEDDING_DIM, 
                window=WINDOW_SIZE, 
                min_count=MIN_COUNT, 
                workers=WORKERS, 
                sg=1
            )
            
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
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean Pooling
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
        return embedding.cpu()

    def _process_biobert_layers(self, encoders: Dict) -> Dict[str, torch.Tensor]:
        """Genera embeddings directamente desde el vocabulario de los encoders."""
        # --- FIX: Asegurar a Pylance que transformers existe ---
        assert AutoTokenizer is not None and AutoModel is not None, "Transformers no está instalado"

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
            bert_dim = model.config.hidden_size 
            
            logger.info(f"Generando embeddings BioBERT para capa: {layer_name} ({vocab_size} items)")
            
            matrix = torch.zeros((vocab_size, bert_dim), dtype=torch.float32)

            for i, entity in tqdm(enumerate(vocab), total=vocab_size, desc=layer_name):
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
        
        if not ENCODER_FILE.exists():
            logger.error(f"No se encuentra el archivo de encoders: {ENCODER_FILE}")
            return

        logger.info(f"Cargando encoders: {ENCODER_FILE}")
        encoders = joblib.load(ENCODER_FILE)

        final_weights = {}

        if self.method == 'GRAPH':
            if KGE_FILE.exists():
                logger.info(f"Cargando embeddings de grafo existentes: {KGE_FILE}")
                kge_wv = gensim.models.KeyedVectors.load_word2vec_format(str(KGE_FILE))
            else:
                G = self._load_graph_data()
                if G.number_of_nodes() == 0:
                    logger.error("Grafo vacío.")
                    return
                kge_wv = self._train_node2vec(G)
            
            final_weights = self._map_graph_embeddings(kge_wv, encoders)

        elif self.method == 'BIOBERT':
            final_weights = self._process_biobert_layers(encoders)

        else:
            logger.error(f"Método desconocido: {self.method}")
            return

        if final_weights:
            logger.info(f"Guardando pesos finales en: {OUTPUT_WEIGHTS_FILE}")
            torch.save(final_weights, OUTPUT_WEIGHTS_FILE)
            
            for k, v in final_weights.items():
                logger.info(f"Tensor guardado -> {k}: {v.shape}")
        else:
            logger.warning("No se generaron pesos. Revisa los logs.")


def main():
    manager = PGxEmbeddingManager(method=EMBEDDING_METHOD)
    manager.run()

if __name__ == "__main__":
    main()