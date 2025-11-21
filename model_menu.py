import argparse
import sys
import logging
from pathlib import Path
import torch
import pandas as pd
from typing import Optional, Dict, Any

# Ajustar path para que encuentre el paquete 'pgen_model' si se ejecuta como script
# (Aunque lo ideal es ejecutar con -m pgen_model)
sys.path.append(str(Path(__file__).parent.parent))

from pgen_model.src.train import train_model
from pgen_model.src.predict import predict_single_input, predict_from_file, load_encoders
from pgen_model.src.model import DeepFM_PGenModel
from pgen_model.src.data import PGenDataProcess, PGenDataset
from pgen_model.src.pipeline import run_training_pipeline
from pgen_model.src.config.config import (
    MODELS_DIR, 
    DATA_DIR, 
    LOGS_DIR, 
    MODEL_CONFIGS, 
    CLINICAL_PRIORITIES,
    MULTI_LABEL_COLUMN_NAMES
)

# Configurar Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(LOGS_DIR) / "pgen_model.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PGenModelCLI")

# --- CACHE GLOBAL ---
_MODEL_CACHE: Dict[str, Any] = {}
_ENCODERS_CACHE: Dict[str, Any] = {}

def get_cached_model(model_name: str, base_dir: Path) -> DeepFM_PGenModel:
    """
    Obtiene el modelo desde la caché o lo carga si no existe.
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    
    logger.info(f"Cargando modelo '{model_name}' en caché...")
    model = load_model(model_name, base_dir)
    _MODEL_CACHE[model_name] = model
    return model

def get_cached_encoders(model_name: str) -> Dict[str, Any]:
    """
    Obtiene los encoders desde la caché o los carga si no existen.
    """
    if model_name in _ENCODERS_CACHE:
        return _ENCODERS_CACHE[model_name]
    
    logger.info(f"Cargando encoders para '{model_name}' en caché...")
    encoders = load_encoders(model_name)
    _ENCODERS_CACHE[model_name] = encoders
    return encoders

def load_model(model_name: str, base_dir: Path) -> DeepFM_PGenModel:
    """
    Carga la arquitectura y pesos del modelo.
    """
    model_path = base_dir / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    # 1. Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 2. Recuperar configuración (guardada en el checkpoint o usar default)
    # Idealmente, guardaríamos la config en el checkpoint. 
    # Si no, asumimos la config del TOML (riesgoso si cambió).
    # Por ahora, intentamos reconstruir con la config actual del TOML.
    
    config = MODEL_CONFIGS["default_model"] # O buscar una específica si hubiera
    
    # Necesitamos saber las dimensiones de features y targets para instanciar
    # Estas suelen venir de los datos.
    # TRUCO: Guardar n_features y target_dims en el checkpoint es vital.
    # Si no están, no podemos reconstruir el modelo dinámico exactamente igual sin los encoders.
    
    if 'model_config' in checkpoint:
        # Si guardaste la config en el entrenamiento (RECOMENDADO)
        m_conf = checkpoint['model_config']
        n_features = m_conf['n_features']
        target_dims = m_conf['target_dims']
        # Params
        emb_dim = m_conf.get('embedding_dim', config['embedding_dim'])
        hidden_dim = m_conf.get('hidden_dim', config['hidden_dim'])
        # ... otros params
    else:
        # Fallback: Intentar inferir o cargar encoders para saber dimensiones
        # Esto es lento y propenso a errores si no tenemos los encoders a mano aquí.
        # Asumiremos que el usuario guardó 'n_features' y 'target_dims' en el checkpoint
        # como se hace en el script de train.py actualizado (si se actualizó).
        # Si no, lanzamos error o warning.
        
        if 'n_features' in checkpoint and 'target_dims' in checkpoint:
            n_features = checkpoint['n_features']
            target_dims = checkpoint['target_dims']
            emb_dim = config['embedding_dim']
            hidden_dim = config['hidden_dim']
        else:
            raise ValueError("El checkpoint no contiene 'n_features' ni 'target_dims'. No se puede reconstruir el modelo.")

    # 3. Instanciar Modelo
    model = DeepFM_PGenModel(
        n_features=n_features,
        target_dims=target_dims,
        embedding_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout_rate=config['dropout_rate'],
        n_layers=config['n_layers'],
        # ... otros argumentos del config
    )
    
    # 4. Cargar Pesos
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Mover a GPU si disponible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model

def predict_one(model_name): # Choice 2 y 3
    print("\n--- Predicción Interactiva ---")
    print("Ingrese los valores (o 'q' para salir):")
    
    try:
        # Usar caché
        encoders = get_cached_encoders(model_name)
        model = get_cached_model(model_name, base_dir=Path(MODELS_DIR))
        
        # Definir columnas objetivo basadas en el modelo (o config)
        # Aquí asumimos todas las del TOML, pero el modelo podría tener menos.
        target_cols = list(model.target_dims.keys())

        while True:
            drug = input("Fármaco (Drug): ").strip()
            if drug.lower() == 'q': break
            
            gene = input("Gen (Gene): ").strip()
            if gene.lower() == 'q': break
            
            allele = input("Alelo (Allele): ").strip()
            if allele.lower() == 'q': break
            
            genotype = input("Genotipo (Variant/Haplotypes): ").strip()
            if genotype.lower() == 'q': break
            
            features = {
                "drug": drug,
                "gene": gene,
                "allele": allele,
                "variant/haplotypes": genotype # Ojo con el nombre, debe coincidir con encoder
            }
            
            print("\nProcesando...")
            results = predict_single_input(features, model, encoders, target_cols)
            
            if results:
                print("\nResultados:")
                for k, v in results.items():
                    print(f"  {k}: {v}")
            else:
                print("Error en la predicción.")
            print("-" * 20)

    except Exception as e:
        logger.error(f"Error en predicción interactiva: {e}")
        print(f"Ocurrió un error: {e}")

def predict_more(model_name): # Choice 4 (Archivo)
    file_path = input("\nIngrese la ruta absoluta del archivo CSV/TSV: ").strip()
    p = Path(file_path)
    
    if not p.exists():
        print("El archivo no existe.")
        return

    try:
        # Usar caché
        encoders = get_cached_encoders(model_name)
        model = get_cached_model(model_name, base_dir=Path(MODELS_DIR))
        target_cols = list(model.target_dims.keys())

        print(f"Procesando archivo: {file_path} ...")
        results = predict_from_file(str(p), model, encoders, target_cols)
        
        if not results:
            print("No se generaron predicciones (verifique logs/errores).")
            return

        # Guardar resultados
        out_path = p.parent / f"{p.stem}_predictions.csv"
        df_res = pd.DataFrame(results)
        df_res.to_csv(out_path, index=False)
        print(f"\nPredicciones guardadas en: {out_path}")
        print(df_res.head())

    except Exception as e:
        logger.error(f"Error en predicción por lotes: {e}")
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="PGenModel CLI")
    parser.add_argument("--mode", choices=["train", "predict", "interactive"], help="Modo de ejecución")
    parser.add_argument("--config", type=str, default="default_model", help="Nombre de la configuración en TOML")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        # Lógica de entrenamiento directa
        run_training_pipeline(config_name=args.config)
        
    elif args.mode == "interactive":
        # Lógica interactiva directa
        predict_one(args.config)
        
    else:
        # Menú Interactivo (Default)
        while True:
            print("\n=== PGenModel Menú Principal ===")
            print("1. Entrenar Modelo (Pipeline Completo)")
            print("2. Predecir (Interactivo - Default Model)")
            print("3. Predecir (Interactivo - Elegir Modelo)")
            print("4. Predecir desde Archivo (Batch)")
            print("5. Salir")
            
            choice = input("Seleccione una opción: ")
            
            if choice == "1":
                conf = input("Nombre de configuración (Enter para 'default_model'): ").strip() or "default_model"
                run_training_pipeline(config_name=conf)
                
            elif choice == "2":
                predict_one("default_model")
                
            elif choice == "3":
                m_name = input("Nombre del modelo (sin .pth): ").strip()
                predict_one(m_name)
                
            elif choice == "4":
                m_name = input("Nombre del modelo a usar (Enter para 'default_model'): ").strip() or "default_model"
                predict_more(m_name)
                
            elif choice == "5":
                print("Saliendo...")
                break
            else:
                print("Opción no válida.")

if __name__ == "__main__":
    main()
