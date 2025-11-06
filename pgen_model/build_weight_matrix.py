import joblib
import gensim
import torch
import sys
from pathlib import Path

# --- 1. Configuración ---
# Asegúrate de que estos nombres de archivo sean correctos.
# El nombre del encoder pkl viene de tu "model_name" en los scripts.
ENCODER_FILE = 'encoders/encoders_Phenotype_Effect_Outcome.pkl'
KGE_FILE = 'encoders/kge_embeddings.kv'
OUTPUT_FILE = 'models/pretrained_weights.pth'

# Las 4 capas de embedding que tu modelo espera,
# deben coincidir con las claves de tu dict de encoders.
EMBEDDING_LAYERS = [
    'drug',
    'variant/haplotypes'
    'gene', 
    'allele', 
    'variant/haplotypes'
]

try:
    # --- 2. Cargar Archivos de Entrada ---
    print(f"Cargando KGE (vectores) desde: {KGE_FILE}")
    kge_wv = gensim.models.KeyedVectors.load_word2vec_format(KGE_FILE)
    embedding_dim = kge_wv.vector_size
    kge_vocab = set(kge_wv.index_to_key) # Un set para búsquedas rápidas
    print(f"KGE cargado. Dimensión del vector: {embedding_dim}")

    print(f"Cargando encoders (vocabulario) desde: {ENCODER_FILE}")
    encoders = joblib.load(ENCODER_FILE)
    print("Encoders cargados.")

    # Diccionario para guardar las matrices de pesos finales
    weight_matrix_dict = {}

    # --- 3. Bucle de "Traducción" ---
    print("\nIniciando 'traducción' de KGE a matrices de pesos...")
    
    total_hits = 0
    total_entities = 0

    for layer_name in EMBEDDING_LAYERS:
        if layer_name not in encoders:
            print(f"Error: No se encontró el encoder para '{layer_name}' en {ENCODER_FILE}", file=sys.stderr)
            continue
        
        # Obtener el vocabulario específico de esta capa (ej. todas las drogas)
        encoder = encoders[layer_name]
        layer_vocab = encoder.classes_
        vocab_size = len(layer_vocab)
        total_entities += vocab_size

        # Crear la matriz de pesos vacía (con ruido aleatorio)
        # Esto es importante para las entidades que NO están en el KGE
        matrix = torch.randn(vocab_size, embedding_dim) * 0.01
        
        hits = 0
        misses = 0
        for i, entity_name in enumerate(layer_vocab):
            # Comprobar si la entidad (ej. "tamoxifen") existe en el KGE
            if entity_name in kge_vocab:
                # Si existe, tomar su vector pre-entrenado
                vector = kge_wv[entity_name]
                matrix[i] = torch.tensor(vector)
                hits += 1
            else:
                # Si no existe, dejamos el ruido aleatorio.
                # El modelo aprenderá este embedding desde cero.
                misses += 1
        
        total_hits += hits
        print(f"  - Capa '{layer_name}':")
        print(f"    - Vocabulario del modelo: {vocab_size} entidades.")
        print(f"    - Entidades encontradas en KGE: {hits} ({hits/vocab_size*100:.2f}%)")
        print(f"    - Entidades no encontradas (se aprenderán): {misses}")
        
        # Guardar la matriz final en el diccionario
        weight_matrix_dict[layer_name] = matrix

    # --- 4. Guardar la Salida ---
    print("\nGuardando diccionario de matrices de pesos en formato PyTorch...")
    torch.save(weight_matrix_dict, OUTPUT_FILE)

    print("\n" + "="*50)
    print(f"¡Éxito! El archivo '{OUTPUT_FILE}' está listo.")
    print(f"Mapeo total: {total_hits} de {total_entities} entidades ({total_hits/total_entities*100:.2f}%)")
    print("Siguiente paso: Modificar model.py y pipeline.py para cargar este archivo.")
    print("="*50)

except FileNotFoundError as e:
    print(f"\n--- ERROR DE ARCHIVO ---", file=sys.stderr)
    print(f"No se pudo encontrar el archivo: {e.filename}", file=sys.stderr)
    print("Asegúrate de que 'kge_embeddings.kv' y tu archivo '.pkl' de encoders están en la misma carpeta.", file=sys.stderr)
except Exception as e:
    print(f"Ocurrió un error inesperado: {e}", file=sys.stderr)