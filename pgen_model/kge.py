import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import gensim
import warnings
import sys

warnings.filterwarnings("ignore", category=FutureWarning)

FILE_PATH = 'train_data/relationships_associated_corrected.tsv'

ENTITY_COLUMNS = ['Entity1_name', 'Entity2_name']

# --- Parámetros de Node2Vec (Hiperparámetros del KGE) ---
EMBEDDING_DIM = 128 # Dimensión de los vectores (128 es un buen inicio)
WALK_LENGTH = 30    # Longitud de cada paseo aleatorio
NUM_WALKS = 200  # Número de paseos aleatorios por nodo

# --- Parámetros de Gensim (Word2Vec) ---
WINDOW_SIZE = 10    # Tamaño de la ventana de contexto
MIN_COUNT = 1       # Contar todas las entidades (incluso las raras)
WORKERS = 16         # Usar 4 cores de CPU para entrenar (ajusta a tu PC)

EMBEDDING_FILE = 'encoders/kge_embeddings.kv' # .kv es el formato de KeyedVectors

print(f"Iniciando el proceso de pre-entrenamiento de embeddings...")
print(f"Archivo de entrada: {FILE_PATH}")
print(f"Dimensiones del embedding: {EMBEDDING_DIM}")

try:
    # --- 2. Cargar Datos y Construir el Grafo ---
    print(f"\nCargando datos desde {FILE_PATH}...")
    df = pd.read_csv(FILE_PATH, sep='\t')
    
    # Filtrar filas donde falte un nombre de entidad
    df = df.dropna(subset=ENTITY_COLUMNS)
    
    for col in ENTITY_COLUMNS:
        df[col] = df[col].astype(str).str.replace(' ', '_')
    
    print("Construyendo el grafo con NetworkX...")
    # Crear un grafo no dirigido a partir de las dos columnas
    G = nx.from_pandas_edgelist(
        df,
        source=ENTITY_COLUMNS[0],
        target=ENTITY_COLUMNS[1],
    )
    
    print(f"Grafo construido exitosamente.")
    print(f"  - Nodos (Entidades Únicas): {G.number_of_nodes()}")
    print(f"  - Aristas (Relaciones): {G.number_of_edges()}")

    # --- 3. Preparar y Ejecutar Node2Vec ---
    # p=1, q=1 hace que se comporte como DeepWalk (paseos sin sesgo)
    print("\nPreparando Node2Vec...")
    n2v = Node2Vec(
        G, 
        dimensions=EMBEDDING_DIM, 
        walk_length=WALK_LENGTH, 
        num_walks=NUM_WALKS, 
        p=1, 
        q=1,
        workers=WORKERS,
        quiet=True
    )

    # --- 4. Entrenar el Modelo (Gensim Word2Vec) ---
    print(f"Generando paseos aleatorios y entrenando el modelo Word2Vec...")
    # .fit() genera los paseos y luego entrena el modelo gensim
    model = n2v.fit(
        window=WINDOW_SIZE, 
        min_count=MIN_COUNT, 
        batch_words=4,
        workers=WORKERS
    )
    print("Entrenamiento del modelo KGE completado.")

    # --- 5. Guardar los Embeddings ---
    model.wv.save_word2vec_format(EMBEDDING_FILE)
    print(f"\n¡Éxito! Embeddings (vectores) guardados en: {EMBEDDING_FILE}")
    print("Este es el archivo que usaremos en el Paso 3.")

    # --- 6. Verificación y Muestra ---
    print("\n--- Verificación de Resultados ---")
    
    # Probar con entidades de tu muestra (si existen en el vocabulario)
    test_entities = ['oxycodone', 'CYP2D6*1', 'Adenocarcinoma', 'PDE4D', 'diuretics']
    
    for entity in test_entities:
        if entity in model.wv:
            print(f"\nEntidades más similares a '{entity}':")
            try:
                similar_nodes = model.wv.most_similar(entity, topn=10)
                for node, similarity in similar_nodes:
                    print(f"  - {node}: {similarity:.4f}")
            except Exception as e:
                print(f"  - Error al buscar similares: {e}")
        else:
            print(f"\nEntidad '{entity}' no encontrada en el vocabulario.")

except FileNotFoundError:
    print(f"Error: No se pudo encontrar el archivo en '{FILE_PATH}'.", file=sys.stderr)
    print("Asegúrate de que 'relationships_associated.csv' está en la misma carpeta.", file=sys.stderr)
except ImportError:
    print("Error: Faltan librerías.", file=sys.stderr)
    print("Asegúrate de haber ejecutado: pip install pandas networkx node2vec gensim", file=sys.stderr)
except KeyError as e:
    print(f"Error de Columna: No se encontró la columna {e} en el CSV.", file=sys.stderr)
    print("Asegúrate de que las columnas se llaman 'Entity1_name' y 'Entity2_name'.", file=sys.stderr)
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}", file=sys.stderr)