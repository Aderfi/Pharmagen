"""
Script para construir matrices de pesos pre-entrenados desde embeddings KGE.

Optimizado para:
- Menor uso de memoria
- Mejor logging de progreso
- Manejo robusto de errores
"""
'''
import joblib
import gensim
import torch
import sys
import logging
from pathlib import Path
from rapidfuzz import process, fuzz
import tqdm
from tqdm import tqdm

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuración ---
# Asegúrate de que estos nombres de archivo sean correctos.
# El nombre del encoder pkl viene de tu "model_name" en los scripts.
ENCODER_FILE = 'encoders/encoders_Phenotype_Effect_Outcome.pkl'
KGE_FILE = 'encoders/kge_embeddings.kv'
OUTPUT_FILE = 'models/pretrained_weights.pth'
THRESHOLD = 95


# Las 4 capas de embedding que tu modelo espera,
# deben coincidir con las claves de tu dict de encoders.
EMBEDDING_LAYERS = [
    'drug',
    'genalle',
    'gene', 
    'allele', 
]

try:
    # --- 2. Cargar Archivos de Entrada ---
    logger.info(f"Cargando KGE (vectores) desde: {KGE_FILE}")
    kge_wv = gensim.models.KeyedVectors.load_word2vec_format(KGE_FILE)
    embedding_dim = kge_wv.vector_size
    kge_vocab = set(kge_wv.index_to_key)  # Un set para búsquedas rápidas
    logger.info(f"KGE cargado. Dimensión del vector: {embedding_dim}, Vocabulario: {len(kge_vocab)} entidades")

    logger.info(f"Cargando encoders (vocabulario) desde: {ENCODER_FILE}")
    encoders = joblib.load(ENCODER_FILE)
    logger.info("Encoders cargados.")

    # Diccionario para guardar las matrices de pesos finales
    weight_matrix_dict = {}

    # --- 3. Bucle de "Traducción" ---
    logger.info("\nIniciando 'traducción' de KGE a matrices de pesos...")
    
    total_hits = 0
    total_entities = 0

    for layer_name in EMBEDDING_LAYERS:
        if layer_name not in encoders:
            logger.error(f"No se encontró el encoder para '{layer_name}' en {ENCODER_FILE}")
            continue
        
        # Obtener el vocabulario específico de esta capa (ej. todas las drogas)
        encoder = encoders[layer_name]
        layer_vocab = encoder.classes_
        vocab_size = len(layer_vocab)
        total_entities += vocab_size

        # Crear la matriz de pesos vacía (con ruido aleatorio)
        # Optimización: usar dtype específico para ahorrar memoria
        matrix = torch.randn(vocab_size, embedding_dim, dtype=torch.float32) * 0.01
        
        hits = 0
        misses = 0
        
        kge_keys = list(kge_wv.key_to_index.keys())
        
        # Optimización: procesar en batch para mejor rendimiento
        for i, entity_name in tqdm(enumerate(layer_vocab), total=len(layer_vocab)):
            
            if entity_name in kge_wv:
                vector = kge_wv[entity_name]
                matrix[i] = torch.tensor(vector, dtype=torch.float32)
                hits += 1
                continue
            
            match_result = process.extractOne(
                entity_name, 
                kge_keys, 
                scorer=fuzz.ratio, 
                score_cutoff=THRESHOLD
            )
            
            if match_result:
                # rapidfuzz devuelve una tupla: (match_string, score, index)
                best_match, score, _ = match_result
                
                # Como ya pasó el cutoff, sabemos que score >= THRESHOLD
                vector = kge_wv[best_match]
                matrix[i] = torch.tensor(vector, dtype=torch.float32)
                hits += 1
                # Opcional: Imprimir qué se arregló para verificar calidad
                # print(f"Fuzzy Match: '{entity_name}' -> '{best_match}' (Score: {score})")
                
            else:
                # 3. NO SE ENCONTRÓ NADA
                misses += 1


         
        
        total_hits += hits
        logger.info(f"  - Capa '{layer_name}':")
        logger.info(f"    - Vocabulario del modelo: {vocab_size} entidades.")
        logger.info(f"    - Entidades encontradas en KGE: {hits} ({hits/vocab_size*100:.2f}%)")
        logger.info(f"    - Entidades no encontradas (se aprenderán): {misses}")
        
        # Guardar la matriz final en el diccionario
        weight_matrix_dict[layer_name] = matrix

    # --- 4. Guardar la Salida ---
    logger.info("\nGuardando diccionario de matrices de pesos en formato PyTorch...")
    # Crear directorio si no existe
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(weight_matrix_dict, OUTPUT_FILE)

    logger.info("\n" + "="*50)
    logger.info(f"¡Éxito! El archivo '{OUTPUT_FILE}' está listo.")
    logger.info(f"Mapeo total: {total_hits} de {total_entities} entidades ({total_hits/total_entities*100:.2f}%)")
    logger.info("Siguiente paso: Modificar model.py y pipeline.py para cargar este archivo.")
    logger.info("="*50)

except FileNotFoundError as e:
    logger.error(f"No se pudo encontrar el archivo: {e.filename}")
    logger.error("Asegúrate de que 'kge_embeddings.kv' y tu archivo '.pkl' de encoders están en la misma carpeta.")
    sys.exit(1)
except Exception as e:
    logger.error(f"Ocurrió un error inesperado: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
    '''
import torch
import gensim
import joblib
import logging
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
from rapidfuzz import process, fuzz

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- 1. Configuración ---
ENCODER_FILE = 'encoders/encoders_Phenotype_Effect_Outcome.pkl'
KGE_FILE = 'encoders/kge_embeddings.kv'
OUTPUT_FILE = 'models/pretrained_weights.pth'
THRESHOLD = 90

EMBEDDING_LAYERS = [
    'drug',
    'genalle',
    'gene', 
    'allele', 
]

try:
    # --- 2. Cargar Archivos de Entrada ---
    logger.info(f"Cargando KGE (vectores) desde: {KGE_FILE}")
    # map_location es importante si se mueve entre CPU/GPU, pero gensim maneja esto bien en CPU
    kge_wv = gensim.models.KeyedVectors.load_word2vec_format(KGE_FILE)
    embedding_dim = kge_wv.vector_size
    
    # OPTIMIZACIÓN: Extraemos las keys UNA sola vez fuera del bucle
    # Esto ahorra memoria y tiempo de procesamiento
    kge_keys = list(kge_wv.key_to_index.keys())
    
    logger.info(f"KGE cargado. Dimensión: {embedding_dim}, Vocabulario KGE: {len(kge_keys)} entidades")

    logger.info(f"Cargando encoders desde: {ENCODER_FILE}")
    encoders = joblib.load(ENCODER_FILE)
    logger.info("Encoders cargados.")

    weight_matrix_dict = {}

    # --- 3. Bucle de "Traducción" ---
    logger.info("\nIniciando mapeo de embeddings...")
    
    total_hits = 0
    total_entities = 0

    for layer_name in EMBEDDING_LAYERS:
        if layer_name not in encoders:
            logger.error(f"No se encontró el encoder para '{layer_name}' en {ENCODER_FILE}")
            continue
        
        # Obtener vocabulario de la capa
        encoder = encoders[layer_name]
        layer_vocab = encoder.classes_
        vocab_size = len(layer_vocab)
        total_entities += vocab_size

        # Inicializar con ruido aleatorio (pequeño)
        matrix = torch.randn(vocab_size, embedding_dim, dtype=torch.float32) * 0.01
        
        hits = 0
        misses = 0
        
        logger.info(f"Procesando capa: {layer_name} ({vocab_size} entidades)...")

        # Iteramos sobre el vocabulario del encoder
        for i, entity_name in tqdm(enumerate(layer_vocab), total=vocab_size):
            
            # PASO 1: Match Exacto (Rápido)
            if entity_name in kge_wv:
                vector = kge_wv[entity_name]
                # .copy() asegura que no haya problemas de memoria negativa con numpy/torch
                matrix[i] = torch.tensor(vector.copy(), dtype=torch.float32)
                hits += 1
                continue
            
            # PASO 2: Match Difuso (Lento, solo si falla el exacto)
            match_result = process.extractOne(
                entity_name, 
                kge_keys, 
                scorer=fuzz.ratio, 
                score_cutoff=THRESHOLD
            )
            
            if match_result:
                # rapidfuzz devuelve (match, score, index)
                best_match, score, _ = match_result
                
                vector = kge_wv[best_match]
                matrix[i] = torch.tensor(vector.copy(), dtype=torch.float32)
                hits += 1
            else:
                # PASO 3: No encontrado (se queda con el ruido aleatorio inicial)
                misses += 1

        total_hits += hits
        
        # Reporte por capa
        logger.info(f"  Resultados '{layer_name}':")
        logger.info(f"    - Encontrados: {hits} ({(hits/vocab_size)*100:.1f}%)")
        logger.info(f"    - No encontrados (Random): {misses}")
        
        weight_matrix_dict[layer_name] = matrix

    # --- 4. Guardar Salida ---
    logger.info("\nGuardando matrices de pesos...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    torch.save(weight_matrix_dict, OUTPUT_FILE)

    logger.info("="*50)
    logger.info(f"¡Proceso completado! Archivo guardado en: {OUTPUT_FILE}")
    if total_entities > 0:
        logger.info(f"Cobertura total: {total_hits}/{total_entities} ({(total_hits/total_entities)*100:.2f}%)")
    logger.info("="*50)

except FileNotFoundError as e:
    logger.error(f"Archivo no encontrado: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Error inesperado: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)