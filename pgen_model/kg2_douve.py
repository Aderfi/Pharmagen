import pandas as pd
import networkx as nx
# Se elimina 'from node2vec import Node2Vec' ya que usamos pecanpy
import gensim
from gensim.models import Word2Vec # Importamos Word2Vec directamente
import warnings
import sys
import multiprocessing
from itertools import product
import os
from pecanpy import pecanpy

warnings.filterwarnings("ignore", category=FutureWarning)

# --- 1. Definir Rutas y Parámetros ---
SNP_DATA_FILE = 'snp_summary.tsv'
DRUG_DATA_FILE = 'drug_gene_edges.tsv'
EMBEDDING_FILE = 'encoders/pgx_embeddings.kv'
OUTPUT_DIR = os.path.dirname(EMBEDDING_FILE)

ENTITY_COLUMNS = ['snp', 'gene', 'Alt_Allele', 'chr', 'clin_sig']

# --- Parámetros de Node2Vec/PecanPy ---
EMBEDDING_DIM = 128
WALK_LENGTH = 30
NUM_WALKS = 150

# --- Parámetros de Gensim (Word2Vec) ---
WINDOW_SIZE = 10
MIN_COUNT = 1
# Usamos WORKERS para PecanPy y Gensim
WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 8))

# --- 2. Funciones de Carga de Datos ---
def load_snp_data(file_path):
    """Carga el archivo TSV de SNPs."""
    print(f"Cargando datos de SNPs desde {file_path}...")
    try:
        snp_df = pd.read_csv(file_path, sep='\t', usecols=ENTITY_COLUMNS)
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

# --- 3. Función de Construcción del Grafo (OPTIMIZADA) ---
def build_graph_fast(snp_df, drug_df):
    """Construye un grafo unificado usando métodos vectorizados de Pandas."""
    print("Construyendo el grafo unificado (modo rápido)...")
    G = nx.Graph()
    
    # --- Parte A: Bordes de SNPs ---
    print("Procesando bordes de SNPs con vectorización...")

    # 1. Separar valores en listas (limpiando nulos antes)
    snp_df['gene'] = snp_df['gene'].fillna('').str.split(';')
    snp_df['Alt_Allele'] = snp_df['Alt_Allele'].fillna('').str.split(',')
    snp_df['clin_sig'] = snp_df['clin_sig'].fillna('').str.split(';')

    # 2. 'Explode' (desenrollar) y añadir bordes
    
    # Bordes SNP <-> Gene
    snp_gene = snp_df[['snp', 'gene']].explode('gene').dropna()
    snp_gene = snp_gene[snp_gene['gene'] != ''] # Excluir strings vacíos
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
    for _, row in gene_clinsig_df.iterrows():
        # Filtra listas vacías o con nulos antes de crear productos
        genes = [g for g in row['gene'] if g]
        sigs = [s for s in row['clin_sig'] if s]
        if genes and sigs:
            edges.extend(list(product(genes, sigs)))
            
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

# --- 4. Función de Entrenamiento y Verificación (OPTIMIZADA) ---
def train_and_verify_fast(G):
    """Entrena el modelo usando PecanPy (rápido) y Gensim."""
    print("\n--- 4. Entrenamiento del Modelo (PecanPy + Gensim) ---")
    if G.number_of_nodes() == 0:
        print("Error: El grafo está vacío, no se puede entrenar.", file=sys.stderr)
        return

    # --- 1. Preparar el grafo para PecanPy ---
    
    # PecanPy funciona con enteros, no con strings.
    # Creamos un mapeo de nuestros nombres de nodo (string) a enteros (int)
    print("Creando mapeo de nodos (String -> Int)...")
    nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(nodes)}
    
    # Mapeo inverso para traducir los paseos de vuelta a strings
    reverse_node_map = {i: node for node, i in node_map.items()}

    # Creamos un grafo NUEVO donde las etiquetas son los enteros
    print("Re-etiquetando el grafo con enteros para C++...")
    G_int = nx.relabel_nodes(G, node_map)

    # Definimos una ruta para el archivo temporal
    TEMP_EDG_FILE = "temp_graph.edg"

    # --- 2. Guardar el grafo en formato Edgelist (archivo de texto) ---
    print(f"Guardando el grafo en el archivo temporal: {TEMP_EDG_FILE}")
    try:
        # Guardamos el grafo de enteros en un formato que PecanPy pueda leer
        # 'data=False' significa que no guardamos pesos, 'delimiter=' '
        nx.write_edgelist(G_int, TEMP_EDG_FILE, data=False, delimiter=' ', encoding='utf-8')
    except Exception as e:
        print(f"Error al escribir el archivo temporal: {e}", file=sys.stderr)
        return

    # --- 3. Ejecutar PecanPy (leyendo desde el archivo) ---
    print("Preparando PecanPy (p=1, q=1)...")
    
    # 3.1. Crear el objeto PecanPy
    g = pecanpy.SparseOTF(
        p=1, 
        q=1, 
        workers=WORKERS, 
        verbose=True
    )

    # 3.2. Cargar el grafo desde el archivo (¡el método correcto!)
    try:
        print("Cargando grafo desde archivo al motor C++ de PecanPy...")
        # 'directed=False' y 'weighted=False'
        g.read_edg(TEMP_EDG_FILE, weighted=False, directed=False, delimiter=' ')
    except Exception as e:
        print(f"Error fatal de PecanPy al leer el archivo .edg: {e}", file=sys.stderr)
        print("Asegúrate de que la biblioteca PecanPy está instalada correctamente.", file=sys.stderr)
        # Limpiar antes de salir
        if os.path.exists(TEMP_EDG_FILE):
             os.remove(TEMP_EDG_FILE)
        return

    # 3.3. Generar los paseos (devuelve listas de ENTEROS)
    print("Generando paseos aleatorios con PecanPy (C++)...")
    int_walks = g.simulate_walks(
        num_walks=NUM_WALKS, 
        walk_length=WALK_LENGTH
    )
    """"""
    # --- 4. Limpiar el archivo temporal ---
    try:
        if os.path.exists(TEMP_EDG_FILE):
            os.remove(TEMP_EDG_FILE)
            print(f"Archivo temporal {TEMP_EDG_FILE} eliminado.")
    except Exception as e:
        print(f"Advertencia: no se pudo eliminar el archivo temporal: {e}", file=sys.stderr)

    # --- 5. Traducir paseos y entrenar Gensim ---
    print("Traduciendo paseos de enteros a strings para Gensim...")
    str_walks = []
    for walk in int_walks:
        # Usamos el mapeo inverso para obtener nuestros nombres originales
        str_walks.append([reverse_node_map[int(node_id)] for node_id in walk])

    print(f"Entrenando el modelo Word2Vec con {WORKERS} workers...")
    model = Word2Vec(
        str_walks, # Usar los paseos de strings
        vector_size=EMBEDDING_DIM,
        window=WINDOW_SIZE,
        min_count=MIN_COUNT,
        workers=WORKERS,
        sg=1 # sg=1 activa el modo Skip-Gram
    )
    
    print("Entrenamiento del modelo KGE completado.")

    # --- 6. Guardar y Verificar (sin cambios) ---
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

# --- 5. Flujo Principal (Main) ---
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
        print("Asegúrate de haber ejecutado: pip install pandas networkx gensim pecanpy", file=sys.stderr)
    except Exception as e:
        print(f"Ocurrió un error inesperado en el flujo principal: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()