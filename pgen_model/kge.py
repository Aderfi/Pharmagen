import pandas as pd
import networkx as nx
from node2vec import Node2Vec
import gensim
import warnings
import sys
import multiprocessing

warnings.filterwarnings("ignore", category=FutureWarning)

FILE_PATH = 'train_data/snp_summary.tsv'

ENTITY_COLUMNS = ['snp', 'gene', 'Alt_Allele', 'chr', 'clin_sig']

# --- Parámetros de Node2Vec (Hiperparámetros del KGE) ---
EMBEDDING_DIM = 256 # Dimensión de los vectores (128 es un buen inicio)
WALK_LENGTH = 30    # Longitud de cada paseo aleatorio
NUM_WALKS = 200  # Número de paseos aleatorios por nodo

# --- Parámetros de Gensim (Word2Vec) ---
WINDOW_SIZE = 10    # Tamaño de la ventana de contexto
MIN_COUNT = 1       # Contar todas las entidades (incluso las raras)
# Optimización: usar número óptimo de workers basado en CPU disponibles
# Dejar 1-2 cores libres para el sistema
WORKERS = max(1, min(multiprocessing.cpu_count() - 2, 8))

EMBEDDING_FILE = 'encoders/rs_chropos_kge_embeddings.kv' # .kv es el formato de KeyedVectors

print(f"Iniciando el proceso de pre-entrenamiento de embeddings...")
print(f"Archivo de entrada: {FILE_PATH}")
print(f"Dimensiones del embedding: {EMBEDDING_DIM}")
print(f"Workers: {WORKERS} (CPU cores disponibles: {multiprocessing.cpu_count()})")

try:
    # --- 2. Cargar Datos y Construir el Grafo ---
    print(f"\nCargando datos desde {FILE_PATH}...")
    # Optimización: especificar dtype y usecols para reducir memoria
    df = pd.read_csv(FILE_PATH, sep='\t', usecols=ENTITY_COLUMNS, 
                     dtype={col: str for col in ENTITY_COLUMNS}, low_memory=False)
    df['gene'].fillna('INTRON', inplace=True)
    
    # Filtrar filas donde falte un nombre de entidad
    df = df.dropna(subset=ENTITY_COLUMNS)
    print(df.columns)
    # Optimización: operación vectorizada en lugar de loop
    # --- CORRECCIÓN INICIO ---
    print("Procesando listas y limpiando nombres...")

    # 1. Convertir las cadenas con comas en listas reales
    # Ejemplo: "GenA, GenB" -> ["GenA", " GenB"]
    for col in ENTITY_COLUMNS[1:3]:  # Solo para 'Alt_Allele' y 'chr'
        # Primero dividir por comas, luego por punto y coma si es necesario
        df[col] = df[col].astype(str).str.split(',')
        df[col] = df[col].astype(str).str.split(';')

    # 2. "Explotar" las listas (Explode)
    # Esto crea una nueva fila por cada elemento de la lista, manteniendo la relación
    # Si tanto Entity1 como Entity2 tienen listas, hacemos explode dos veces para cubrir todas las combinaciones (producto cartesiano)
    df = df.explode(ENTITY_COLUMNS[1]).explode(ENTITY_COLUMNS[2])

    # 3. Limpieza final: Quitar espacios en blanco sobrantes y sustituir espacios internos por guion bajo
    # Es importante hacer el strip() primero por si quedaron espacios tras la coma (ej: "GenA, GenB")
    for col in ENTITY_COLUMNS:
        df[col] = df[col].str.strip().str.replace(' ', '_')

    # 4. Eliminar posibles vacíos generados por comas consecutivas o finales
    df = df.dropna(subset=ENTITY_COLUMNS)
    df = df[(df[ENTITY_COLUMNS[1]] != '') & (df[ENTITY_COLUMNS[2]] != '')]
    # --- CORRECCIÓN FIN ---
    
    print("Construyendo el grafo con NetworkX...")
    # Crear un grafo no dirigido a partir de las dos columnas
    G = nx.from_pandas_edgelist(
        df,
        source=ENTITY_COLUMNS[0],
        target=ENTITY_COLUMNS[1],
        edge_attr=ENTITY_COLUMNS[2]
    )
    
    print(f"Grafo construido exitosamente.")
    print(f"  - Nodos (Entidades Únicas): {G.number_of_nodes()}")
    print(f"  - Aristas (Relaciones): {G.number_of_edges()}")

    # --- 3. Preparar y Ejecutar Node2Vec ---
    # p=1, q=1 hace que se comporte como DeepWalk (paseos sin sesgo)
    # ENTITY_COLUMNS = ['snp', 'gene', 'Alt_Allele', 'chr', 'pos']
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