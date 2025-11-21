import pandas as pd
#import polars as pl # Uncomment and use pl.read_csv for very large files!
import networkx as nx
import gensim
from gensim.models import Word2Vec
import warnings
import sys
import multiprocessing
from itertools import product
import os
from pecanpy import pecanpy
from tqdm import tqdm  # Add tqdm for optional progress bars

warnings.filterwarnings("ignore", category=FutureWarning)

SNP_DATA_FILE = 'snp_summary.tsv'
DRUG_DATA_FILE = 'drug_gene_edges.tsv'
EMBEDDING_FILE = 'encoders/pgx_embeddings.kv'
OUTPUT_DIR = os.path.dirname(EMBEDDING_FILE)

ENTITY_COLUMNS = ['snp', 'gene', 'Alt_Allele', 'chr', 'clin_sig']

EMBEDDING_DIM = 128
WALK_LENGTH = 30
NUM_WALKS = 150
WINDOW_SIZE = 10
MIN_COUNT = 1
WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 8))

def load_snp_data(file_path):
    """Carga el archivo TSV de SNPs."""
    print(f"Cargando datos de SNPs desde {file_path}...")
    try:
        snp_df = pd.read_csv(file_path, sep='\t', usecols=ENTITY_COLUMNS)
        # For very big files:
        # snp_df = pl.read_csv(file_path, separator='\t').to_pandas()
        snp_df.columns = snp_df.columns.str.strip()
        print("Datos de SNPs cargados.")
        return snp_df
    except FileNotFoundError:
        print(f"Error: No se encontró {file_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def load_drug_data(file_path):
    """Carga el archivo CSV de Fármaco-Gen."""
    print(f"Cargando datos de Fármacos desde {file_path}...")
    try:
        drug_df = pd.read_csv(file_path, sep='\t')
        print("Datos de Fármacos cargados.")
        return drug_df
    except FileNotFoundError:
        print(f"Error: No se encontró {file_path}", file=sys.stderr)
        print("Asegúrate de tener el archivo de relaciones fármaco-gen.")
        sys.exit(1)
    except Exception as e:
        print(f"Error al leer {file_path}: {e}", file=sys.stderr)
        sys.exit(1)

def build_graph_fast(snp_df, drug_df):
    """Construye un grafo unificado usando métodos vectorizados de Pandas."""
    print("Construyendo el grafo unificado (modo rápido)...")
    G = nx.Graph()
    
    # --- Parte A: Bordes de SNPs ---
    print("Procesando bordes de SNPs con vectorización...")

    snp_df['gene'] = snp_df['gene'].fillna('').str.split(';')
    snp_df['Alt_Allele'] = snp_df['Alt_Allele'].fillna('').str.split(',')
    snp_df['clin_sig'] = snp_df['clin_sig'].fillna('').str.split(';')

    # Bordes SNP <-> Gene
    snp_gene = snp_df[['snp', 'gene']].explode('gene').dropna()
    snp_gene = snp_gene[snp_gene['gene'] != '']
    G.add_edges_from(snp_gene.itertuples(index=False), relationship='snp_to_gene')
    print(f"Añadidos {len(snp_gene)} bordes snp_to_gene.")

    # Bordes SNP <-> Allele
    snp_allele = snp_df[['snp', 'Alt_Allele']].explode('Alt_Allele').dropna()
    snp_allele = snp_allele[snp_allele['Alt_Allele'] != '']
    G.add_edges_from(snp_allele.itertuples(index=False), relationship='snp_to_allele')
    print(f"Añadidos {len(snp_allele)} bordes snp_to_allele.")

    # Bordes SNP <-> Clinical Significance
    snp_clinsig = snp_df[['snp', 'clin_sig']].explode('clin_sig').dropna()
    snp_clinsig = snp_clinsig[snp_clinsig['clin_sig'] != '']
    G.add_edges_from(snp_clinsig.itertuples(index=False), relationship='snp_to_clin_sig')
    print(f"Añadidos {len(snp_clinsig)} bordes snp_to_clin_sig.")

    # Bordes Gene <-> Clinical Significance (Producto cartesiano)
    gene_clinsig_df = snp_df[['gene', 'clin_sig']].dropna()
    edges = []
    # Use itertuples for faster iteration
    for row in tqdm(gene_clinsig_df.itertuples(index=False), desc="gene_to_clin_sig", total=len(gene_clinsig_df)):
        genes = [g for g in row.gene if g]
        sigs = [s for s in row.clin_sig if s]
        if genes and sigs:
            # Use generator instead of list + extend for less memory if necessary
            edges.extend(product(genes, sigs))
    G.add_edges_from(edges, relationship='gene_to_clin_sig')
    print(f"Añadidos {len(edges)} bordes gene_to_clin_sig.")

    # --- Parte B: Bordes de Fármacos ---
    print("Añadiendo bordes de Fármaco-Gen...")
    drug_edges = drug_df[['drug', 'gene']].dropna()
    G.add_edges_from(drug_edges.itertuples(index=False), relationship='drug_to_gene')
    print(f"Añadidos {len(drug_edges)} bordes drug_to_gene.")
    
    # --- Resumen del Grafo Final ---
    print("\n--- Resumen del Grafo Unificado ---")
    print(f"Nodos Totales: {G.number_of_nodes()}")
    print(f"Bordes Totales: {G.number_of_edges()}")
    
    return G

def train_and_verify_fast(G):
    """Entrena el modelo usando PecanPy (rápido) y Gensim."""
    print("\n--- 4. Entrenamiento del Modelo (PecanPy + Gensim) ---")
    if G.number_of_nodes() == 0:
        print("Error: El grafo está vacío, no se puede entrenar.", file=sys.stderr)
        return

    print("Creando mapeo de nodos (String -> Int)...")
    nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    reverse_node_map = {i: node for node, i in node_map.items()}
    print("Re-etiquetando el grafo con enteros para C++...")
    G_int = nx.relabel_nodes(G, node_map)

    TEMP_EDG_FILE = "temp_graph.edg"

    # Write once, read once, then (if PecanPy allows) migrate to in-memory loading in future PecanPy versions.
    print(f"Guardando el grafo en el archivo temporal: {TEMP_EDG_FILE}")
    try:
        nx.write_edgelist(G_int, TEMP_EDG_FILE, data=False, delimiter=' ', encoding='utf-8')
    except Exception as e:
        print(f"Error al escribir el archivo temporal: {e}", file=sys.stderr)
        return

    print("Preparando PecanPy (p=1, q=1)...")
    g = pecanpy.SparseOTF(
        p=1, 
        q=1, 
        workers=WORKERS, 
        verbose=True
    )

    try:
        print("Cargando grafo desde archivo al motor C++ de PecanPy...")
        g.read_edg(TEMP_EDG_FILE, weighted=False, directed=False, delimiter=' ')
    except Exception as e:
        print(f"Error fatal de PecanPy al leer el archivo .edg: {e}", file=sys.stderr)
        if os.path.exists(TEMP_EDG_FILE):
             os.remove(TEMP_EDG_FILE)
        return

    print("Generando paseos aleatorios con PecanPy (C++)...")
    int_walks = g.simulate_walks(
        num_walks=NUM_WALKS, 
        walk_length=WALK_LENGTH
    )

    try:
        if os.path.exists(TEMP_EDG_FILE):
            os.remove(TEMP_EDG_FILE)
            print(f"Archivo temporal {TEMP_EDG_FILE} eliminado.")
    except Exception as e:
        print(f"Advertencia: no se pudo eliminar el archivo temporal: {e}", file=sys.stderr)

    print("Traduciendo paseos de enteros a strings para Gensim...")
    # If enough RAM, list comp will be faster. tqdm is optional, for very large sets of walks
    str_walks = [
        [reverse_node_map[int(node_id)] for node_id in walk]
        for walk in tqdm(int_walks, desc="translate_walks")
    ]

    print(f"Entrenando el modelo Word2Vec con {WORKERS} workers...")
    model = Word2Vec(
        str_walks,
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1
    )
    
    print("Entrenamiento del modelo KGE completado.")

    print("\n--- 5. Guardando Embeddings ---")
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Directorio creado: {OUTPUT_DIR}")
        
        model.wv.save_word2vec_format(EMBEDDING_FILE)
        print(f"¡Éxito! Embeddings guardados en: {EMBEDDING_FILE}")
    except Exception as e:
        print(f"Error al guardar el archivo de embeddings: {e}", file=sys.stderr)

    print("\n--- 6. Verificación de Resultados ---")
    test_entities = ['Warfarin', 'Rosuvastatin', 'APOE', 'rs671', 'drug-response', 'AGT']
    
    for entity in test_entities:
        if entity in model.wv:
            print(f"\nEntidades más similares a '{entity}':")
            try:
                similar_nodes = model.wv.most_similar(entity, topn=10)
                for node, similarity in similar_nodes:
                    print(f"   - {node}: {similarity:.4f}")
            except Exception as e:
                print(f"   - Error al buscar similares: {e}")
        else:
            print(f"\nEntidad '{entity}' no encontrada en el vocabulario.")

def main():
    """Función principal que orquesta el proceso."""
    print(f"Iniciando el proceso de pre-entrenamiento de embeddings PGx (Optimizado)...")
    print(f"Archivos de entrada: {SNP_DATA_FILE} y {DRUG_DATA_FILE}")
    print(f"Dimensiones del embedding: {EMBEDDING_DIM}")
    print(f"Workers: {WORKERS} (CPU cores disponibles: {multiprocessing.cpu_count()})\n")

    try:
        # Cargar datos
        snp_df = load_snp_data(SNP_DATA_FILE)
        drug_df = load_drug_data(DRUG_DATA_FILE)
        
        # Construir grafo (usando la función rápida)
        G = build_graph_fast(snp_df, drug_df)
        
        # Entrenar y verificar (usando la función rápida)
        train_and_verify_fast(G)
        
        print("\nProceso finalizado.")

    except ImportError:
        print("Error: Faltan librerías.", file=sys.stderr)
        print("Asegúrate de haber ejecutado: pip install pandas networkx gensim pecanpy tqdm", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error inesperado en el flujo principal: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()