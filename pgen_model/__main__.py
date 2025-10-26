"""
Punto de entrada para el paquete pgen_model.
Proporciona un menú CLI para entrenar y utilizar el modelo predictivo.
"""
import json
import math
import sys
from pathlib import Path

from src.config.config import MODEL_TRAIN_DATA, PGEN_MODEL_DIR

from .data import PGenInputDataset, train_data_import
from .model import DeepFM_PGenModel
from .model_configs import MODEL_REGISTRY, get_model_config
from .pipeline import train_pipeline
from .predict import load_encoders, predict_from_file, predict_single_input

###########################################################
# Configuración de Optuna
#N_TRIALS = 50  # Número de pruebas para Optuna
PGEN_MODEL_DIR = "."  # Ajusta si tu variable global es distinta
###########################################################

def select_model(model_options, prompt="Selecciona el modelo:"):
    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(model_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    model_choice = ""
    while model_choice not in [str(i+1) for i in range(len(model_options))]:
        model_choice = input(f"{prompt} (1-{len(model_options)}): ").strip()
        if model_choice not in [str(i+1) for i in range(len(model_options))]:
            print("Opción no válida. Intente de nuevo.")
    return model_options[int(model_choice)-1]

def load_model(model_name, target_cols=None, base_dir=None, device=None):
    """
    Carga un modelo PyTorch guardado para un conjunto específico de targets.
    
    Args:
        model_name (str): Nombre del modelo a cargar
        target_cols (list, optional): Lista de columnas target. Si es None, 
                                     se obtienen del MODEL_REGISTRY
        base_dir (str, optional): Directorio base donde buscar. Si es None,
                                 usa el directorio de modelos por defecto
        device (torch.device, optional): Dispositivo donde cargar el modelo
                                        (CPU o GPU)
    
    Returns:
        torch.nn.Module: El modelo cargado y listo para usar
    """
    from pathlib import Path

    import torch

    from .model import DeepFM_PGenModel
    from .model_configs import MODEL_REGISTRY
    
    if base_dir is None:
        # Usar el mismo directorio que en pipeline.py
        from src.config.config import PGEN_MODEL_DIR as MODELS_DIR
        base_dir = MODELS_DIR
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # Si no se proporcionan target_cols, obtenerlos del registro de modelos
    if target_cols is None:
        target_cols = [t.lower() for t in MODEL_REGISTRY[model_name]['targets']]
    
    # Crear el nombre del archivo del modelo siguiendo el formato en train.py
    model_file_name = str('-'.join([target[:3] for target in target_cols]))
    model_file = Path(base_dir) / f'pmodel_{model_file_name}.pth'
    
    if not model_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_file}")
    
    try:
        # Cargar los encoders para obtener las dimensiones necesarias
        encoders = load_encoders(model_name, base_dir)
        
        # Obtener las dimensiones necesarias para reconstruir el modelo
        n_drugs = len(encoders['drug'].classes_)
        n_genes = len(encoders['gene'].classes_)
        n_alleles = len(encoders['allele'].classes_)
        n_genotypes = len(encoders['genotype'].classes_)
        
        # Determinar las dimensiones de salida para cada target
        target_dims = {}
        for col in target_cols:
            target_dims[col] = len(encoders[col].classes_)
        
        # Obtener los hiperparámetros del modelo desde el registro
        params = MODEL_REGISTRY[model_name].get('params', {})
        embedding_dim = params.get('embedding_dim', 256)
        hidden_dim = params.get('hidden_dim', 512)
        dropout_rate = params.get('dropout_rate', 0.3)
        
        # Crear una instancia del modelo con la arquitectura correcta
        model = DeepFM_PGenModel(
            n_drugs, n_genes, n_genotypes, #n_alleles,
            embedding_dim, hidden_dim, dropout_rate,
            target_dims=target_dims
        )
        
        # Cargar el state_dict y asignarlo al modelo
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        
        # Mover el modelo al dispositivo especificado y ponerlo en modo evaluación
        model = model.to(device)
        model.eval()
        
        print(f"Modelo cargado correctamente desde: {model_file}")
        return model
    
    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")
    
    
def main():
    # --- CAMBIO: Usar MODEL_REGISTRY ---
    model_options = list(MODEL_REGISTRY.keys()) 

    while True:
        print("""
                                                                ———————————————— Pharmagen PModel ————————————————
                                                                |                                                 |
                                                                |  1. Entrenar modelo                             |    
                                                                |                                                 |
                                                                |  2. Realizar predicción (datos manuales)        |
                                                                |                                                 |
                                                                |  3. Realizar predicción (desde archivo)         |
                                                                |                                                 |
                                                                |  4. Optimizacion de hiperparámetros (Optuna)    |
                                                                |                                                 |
                                                                |           5. Salir                              |
                                                                ——————————————————————————————————————————————————
""")
        choice = input("Selecciona opción (1-5): ").strip()

        # =====================  1: ENTRENAMIENTO DEL MODELO --> train.py  =======================================

        if choice == "1":

            model_name = select_model(model_options, "Selecciona el modelo para entrenamiento")
            
            config = get_model_config(model_name) 
            params = config['params'] # Diccionario con HPs (LR, HD, etc.) 
            target_cols = [t.lower() for t in config['targets']] 
            
            epochs = params.get('epochs', 100)
            patience = params.get('patience', 15)
            batch_size = params.get('batch_size', 64)

            PMODEL_DIR = "pgen_model/models/" 
            csv_files = Path(Path(PMODEL_DIR) / 'train_data' / 'train_therapeutic_outcome.csv')
            
            print(f"Iniciando entrenamiento con modelo: {model_name}")
            print(f"Parámetros: {json.dumps(params, indent=2)}")
            
            # 3. Llamar a pipeline
            train_pipeline(
                PMODEL_DIR, 
                csv_files, 
                model_name, 
                params, 
                epochs=epochs, 
                patience=patience, 
                batch_size=batch_size, 
                target_cols=target_cols
            )

        # =====================================  2: PREDICCIÓN MANUAL ============================================

        elif choice == "2":
            model_name = select_model(model_options, "Selecciona el modelo para predicción manual")
            targets = MODEL_REGISTRY[model_name]['targets'] 
            target_cols = [t.lower() for t in targets]
            
            print("Introduce datos del paciente para predicción:")
            drug = input("Drug: ")
            gene = input("Gene: ")
            allele = input("Allele: ")
            genotype = input("Genotype: ")
            
            try:
                encoders = load_encoders(model_name)
                
                model = load_model(model_name, target_cols=target_cols)
                
                resultado = predict_single_input(drug, gene, allele, genotype, model=model, encoders=encoders, target_cols=target_cols)
                
                print("\nResultado de la predicción:")
                if resultado is not None:
                    for k, v in resultado.items():
                        print(f"{k}: {v}")
                else:
                    print("No se pudo realizar la predicción.")
            except Exception as e:
                print(f"Error al cargar modelo o encoders: {e}")

        # =====================================  3: PREDICCIÓN DESDE ARCHIVO ============================================

        elif choice == "3":
            model_name = select_model(model_options, "Selecciona el modelo para predicción desde archivo")
            # --- CAMBIO: Usar MODEL_REGISTRY ---
            targets = MODEL_REGISTRY[model_name]['targets'] 
            target_cols = [t.lower() for t in targets]
            file_path = input("Ruta del archivo CSV: ")
            try:
                print("ADVERTENCIA: Debes implementar la carga del modelo y los encoders.")
                # Carga el modelo y encoders según tu flujo aquí...
                # model, encoders = ...
                # resultados = predict_from_file(file_path, model=model, encoders=encoders, target_cols=target_cols)
                resultados = [] 
                print("\nResultados de las predicciones:")
                print(resultados)
            except Exception as e:
                print(f"Error procesando archivo: {e}")

        # =====================================  4: OPTIMIZACIÓN DE HIPERPARÁMETROS ============================================

        elif choice == "4":
            import optuna

            from .optuna_train import run_optuna_with_progress
            print("\nOptimizando hiperparámetros con Optuna...")
            optuna_model_name = select_model(model_options, "Selecciona el modelo para optimización")
            
            # --- CAMBIO: Usar MODEL_REGISTRY ---
            targets = MODEL_REGISTRY[optuna_model_name]['targets'] 
            target_cols = [t.lower() for t in targets]
            
            best_params, best_loss, results_file, normalized_loss = run_optuna_with_progress(
                optuna_model_name,
                output_dir=Path(PGEN_MODEL_DIR),
                target_cols=target_cols
            )

            print(f"\nMejores hiperparámetros encontrados ({optuna_model_name}):")
            print(json.dumps(best_params, indent=2))
            print("Mejor loss:", best_loss)

            print(f"\nPérdida normalizada del mejor modelo: {normalized_loss:.4f} (Valor máximo 0, mínimo 1)")
            print(f"Top 5 trials guardados en: {results_file}")

        elif choice == "5":
            print("¡Gracias por usar Pharmagen PModel!")
            sys.exit(0)
         
        # ===================   Opción oculta   ==============================    
        # Obtener estimación inicial de weights (diagnóstico tras 1 epoch)
        elif choice == "777":
        

            model_name = select_model(model_options, "Selecciona el modelo para diagnóstico de pesos")

            config = get_model_config(model_name)
            try:
                params = dict(config['params']) # Diccionario con HPs (LR, HD, etc.)
            except KeyError:
                print(f"Error: No se encontraron los parámetros para el modelo {model_name}")
                continue
            target_cols = [t.lower() for t in config['targets']]

            # --- CAMBIO: Ejecutar SOLO 1 época para el diagnóstico ---
            epochs = 1
            patience = 1 # No necesitamos early stopping para 1 época
            batch_size = params.get('batch_size', 64)

            # --- CAMBIO: Asegurarse de que los pesos son 1.0 para el diagnóstico ---
            # (Esto asume que tu get_model_config ya devuelve los pesos correctos)
            # Si quieres forzar pesos=1.0 *aquí*, necesitarías modificar cómo
            # train_pipeline obtiene los pesos, o pasar un flag diferente.
            # Por ahora, asumimos que los pesos en model_configs.py son 1.0
            # o que train_pipeline los ignora si es un run de diagnóstico.
            # La forma más limpia es ponerlos a 1.0 en model_configs.py temporalmente.
            print("ADVERTENCIA: Asegúrate de que los MASTER_WEIGHTS en model_configs.py están en 1.0 para este diagnóstico.")

            csv_files = Path(MODEL_TRAIN_DATA / 'train_therapeutic_outcome.csv')

            
            print(f"Iniciando run de diagnóstico (1 época) con modelo: {model_name}")
            print(f"Parámetros: {', '.join(f'{k}: {v}' for k, v in params.items())} ")  # No imprimir weights aquí

            
            train_pipeline(
                PGEN_MODEL_DIR,
                csv_files,
                model_name,
                params,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
            
            )
            print("\nRun de diagnóstico completado. Revisa 'comprobacion.txt' o los logs para ver los losses individuales.")

if __name__ == "__main__":
    main()