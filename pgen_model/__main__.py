"""
Punto de entrada para el paquete pgen_model.
Proporciona un menú CLI para entrenar y utilizar el modelo predictivo.
"""

import json
from logging import Logger
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from tabulate import tabulate

from src.scripts.utils import _search_files
from pgen_model.src.data import PGenDataProcess
from pgen_model.src.model import DeepFM_PGenModel
from pgen_model.src.model_configs import MODEL_REGISTRY, get_model_config
from pgen_model.src.predict import load_encoders, predict_from_file, predict_single_input
from pgen_model.src.pipeline import train_pipeline

from src.config.config import (
    MODEL_TRAIN_DATA, 
    PGEN_MODEL_DIR, 
    PROJECT_ROOT, 
    MODELS_DIR,
    PREDICT_FILES_DIR,
    DATA_DIR
)

###########################################################
# Configuración
PGEN_MODEL_DIR = "." 
###########################################################
logger = Logger("pgen_model_main")


translate_output: Dict[str, str] = {
    "phenotype_outcome": "Consecuencia clínica esperable",
    "effect_direction": "Variación del efecto farmacogenético",
    "effect_type": "Tipo de efecto farmacogenético",
    "effect_phenotype": "Tipo de efecto - Extensión",
    "Fenotipo_No_Especificado": ""
}

def select_model(model_options: List[str], prompt: str = "Selecciona el modelo:") -> str:
    """
    Presenta un menú interactivo para seleccionar un modelo.
    
    Args:
        model_options: Lista de nombres de modelos disponibles
        prompt: Mensaje a mostrar al usuario
        
    Returns:
        Nombre del modelo seleccionado
    """
    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(model_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    model_choice = ""
    valid_choices = [str(i) for i in range(1, len(model_options) + 1)]
    while model_choice not in valid_choices:
        model_choice = input(f"{prompt} (1-{len(model_options)}): ").strip()
        if model_choice not in valid_choices:
            print("Opción no válida. Intente de nuevo.")
    return model_options[int(model_choice) - 1]

def load_model(
    model_name: str,
    base_dir: Optional[Path] = None,
    device: Optional[torch.device] = None
) -> DeepFM_PGenModel:
    """
    Carga un modelo PyTorch guardado para un conjunto específico de targets.
    
    Args:
        model_name: Nombre del modelo a cargar
        target_cols: Lista de columnas objetivo (se obtiene de config si es None)
        base_dir: Directorio base donde están los modelos (usa MODELS_DIR si es None)
        device: Dispositivo PyTorch (auto-detecta si es None)
        
    Returns:
        Modelo cargado y listo para inferencia
        
    Raises:
        FileNotFoundError: Si el archivo del modelo no existe
        Exception: Si hay error al cargar el modelo
    """
    if base_dir is None:
        base_dir = MODELS_DIR

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_cols = [f.lower() for f in MODEL_REGISTRY[model_name]["features"]]
    target_cols = [t.lower() for t in MODEL_REGISTRY[model_name]["targets"]]

    model_file = Path(base_dir) / f"pmodel_{model_name}.pth"

    if not model_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo del modelo en {model_file}")

    try:
        encoders = load_encoders(model_name)

        feature_dims = {col: len(encoders[col].classes_) for col in feature_cols}
        target_dims = {col: len(encoders[col].classes_) for col in target_cols}

        # Obtener los hiperparámetros del modelo
        config = get_model_config(model_name)
        params = config["params"]
        embedding_dim = params.get("embedding_dim", 256)
        hidden_dim = params.get("hidden_dim", 512)
        dropout_rate = params.get("dropout_rate", 0.3)
        n_layers = params.get("n_layers", 2)


        model = DeepFM_PGenModel(
        n_features=feature_dims,              
        target_dims=target_dims,
        embedding_dim=params["embedding_dim"],
        n_layers=params["n_layers"],
        hidden_dim=params["hidden_dim"],
        dropout_rate=params["dropout_rate"],
        attention_dim_feedforward=params.get("attention_dim_feedforward"),
        attention_dropout=params.get("attention_dropout", 0.1),
        num_attention_layers=params.get("num_attention_layers", 1),
        use_batch_norm=params.get("use_batch_norm", False),
        use_layer_norm=params.get("use_layer_norm", False),
        activation_function=params.get("activation_function", "gelu"),
        fm_dropout=params.get("fm_dropout", 0.1),
        fm_hidden_layers=params.get("fm_hidden_layers", 0),
        fm_hidden_dim=params.get("fm_hidden_dim", 256),
        embedding_dropout=params.get("embedding_dropout", 0.1),
        separate_embedding_dims=None,
    ).to(device)

        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()

        print(f"Modelo cargado correctamente desde: {model_file}")
        return model

    except Exception as e:
        raise Exception(f"Error al cargar el modelo: {e}")

def cli_menu(model_name) -> int:
    while True:
        print(
            """
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
        """
        )
        choice: int = int(input(f"Selecciona opción (1-5): "))
        if choice not in range(1, 5):
            print("Opción no válida. Intente de nuevo.")
            continue
        else:
            if choice == 1:
                train_choice(model_name)
            elif choice == 2:
                predict_one(model_name)
            elif choice ==3:
                predict_more(model_name)
            elif choice ==4:
                optuna_choice(model_name)
            elif choice == 777:
                _get_model_autoweights(model_name)
            elif choice == 5:
                print("Saliendo del programa.")
                quit()

def train_choice(model_name): # Choice 1
    config = get_model_config(model_name)
    params = config["params"]  # Diccionario con HPs
            
    # Extraer args para pipeline desde los params
    epochs = params.get("epochs", 100)
    patience = params.get("patience", 20)
    batch_size = params.get("batch_size", 64)
    target_cols = [t.lower() for t in config["targets"]]
            
    csv_files = Path(MODEL_TRAIN_DATA, "final_test_genalle.tsv")
    PMODEL_DIR = PROJECT_ROOT # PGEN_MODEL_DIR se define arriba como "."

    print(f"Iniciando entrenamiento con modelo: {model_name}")
    print(f"Parámetros: {json.dumps(params, indent=2)}")

    # <--- CORRECCIÓN 2: Llamada a train_pipeline ---
    train_pipeline(
        PMODEL_DIR,
        csv_files,
        model_name,
        target_cols=target_cols,
    )        
    
def optuna_choice(model_name): # Choice 4
        import optuna
        from pgen_model.src.optuna_train import run_optuna_with_progress

        print("\nOptimizando hiperparámetros con Optuna...")

        features = MODEL_REGISTRY[model_name]["features"]
        targets = MODEL_REGISTRY[model_name]["targets"]
        feature_cols = [f.lower() for f in features]
        target_cols = [t.lower() for t in targets]
        
        print("\n¿Qué tipo de optimización deseas usar?")
        print("  1. Multi-Objetivo (Loss + F1) [RECOMENDADO para Pharmacogenomics]")
        print("  2. Single-Objetivo (Solo Loss)")
        opt_mode = input("Selecciona (1-2): ").strip()

        use_multi_obj = (opt_mode == "1")
            
        best_params, best_loss, normalized_loss = (
            run_optuna_with_progress(
                model_name=model_name,
                output_dir=Path(PGEN_MODEL_DIR),
                use_multi_objective=use_multi_obj,
            )
        )

        logger.info(f"Entrenamiento de optuna finalizado para {model_name}")
    
def predict_one(model_name): # Choice 2 y 3
    csv_path = Path(MODEL_TRAIN_DATA)
    predict_files_path = Path(PREDICT_FILES_DIR)
    
    predict_files: list[str] = _search_files(predict_files_path, ext=None)
    
    
    targets = MODEL_REGISTRY[model_name]["targets"]
    target_cols = [t.lower() for t in targets]

    print("Introduce datos del paciente para predicción:")
    features: List[str] = MODEL_REGISTRY[model_name]["features"] # type: ignore
    features_dict: Dict[str, str] = {feature.lower(): input(f"{feature}: ") for feature in features}

    try:
        encoders = load_encoders(model_name) 
        model = load_model(
                model_name, 
                base_dir=MODELS_DIR
            )

        
        resultado = predict_single_input(
            features_dict,
            model=model,
            encoders=encoders,
            target_cols=target_cols,
        )
        drug = features_dict.get("drug", "N/A").capitalize()
        genotype = features_dict.get("genotype", "N/A").capitalize()
        gene = features_dict.get("gene", "N/A").capitalize()
        allele = features_dict.get("allele", "N/A").capitalize()
        
        resultado_df = pd.DataFrame({"Medicamento": [drug], "Variant/Haplotypes": [genotype], "Gene": [gene], "Allele": [allele]})
        resultado_df = pd.concat([resultado_df, pd.DataFrame(resultado)], axis=1)
        
        '''
        try:
            # MAPEO THERAPEUTIC_OUTCOME
            json_path = Path(MODEL_TRAIN_DATA, "json_dicts", "dict_therapeutic_outcome.json")
            
            with open(json_path, "r", encoding="utf-8") as f:
                ther_out_dict = json.load(f)
                ther_out_dict = {k.lower(): v for k, v in ther_out_dict.items()}
            
            # Tomar solo los primeros 3 valores del diccionario resultado
            primeros_valores = list(resultado.values())  # type: ignore
            
            # Crear la línea formateada
            linea = ';'.join(
                ','.join(map(str, v)) if isinstance(v, list) else str(v) 
                for v in primeros_valores
            )
            
            # Buscar el valor correspondiente (case-insensitive)
            therapeutic_outcome = ther_out_dict.get(linea.lower(), "nan")
            resultado_df['Therapeutic_Outcome'] = therapeutic_outcome
            
            print(f"Línea de búsqueda: {linea}")
            print(f"Resultado encontrado: {therapeutic_outcome}")

        except Exception as e:
            print(f"Error al mapear Therapeutic_Outcome: {e}")
            resultado_df['Therapeutic_Outcome'] = "nan"
        '''


        print("\nResultado de la predicción:")
        if resultado is not None:
            print(tabulate(resultado_df, headers='keys', tablefmt='psql', showindex=False)) # type: ignore

            print("Pulse cualquier tecla para continuar...")
            input()

        else:
            print("No se pudo realizar la predicción.")
    except Exception as e:
        print(f"Error al cargar modelo o encoders: {e}")

def predict_more(model_name): # Choice 3
    targets = MODEL_REGISTRY[model_name]["targets"]
    target_cols = [t.lower() for t in targets]
    file_path = input("Ruta del archivo CSV: ")
    try:
        print(
            "ADVERTENCIA: Debes implementar la carga del modelo y los encoders."
        )
        # Carga el modelo y encoders según tu flujo aquí...
        # model, encoders = ...
        # resultados = predict_from_file(file_path, model=model, encoders=encoders, target_cols=target_cols)
        resultados = []
        print("\nResultados de las predicciones:")
        print(resultados)
    except Exception as e:
        print(f"Error procesando archivo: {e}")

def _get_model_autoweights(model_name): # Choice 777
    config = get_model_config(model_name)
    try:
        params = dict(config["params"])
    except KeyError:
        print(f"Error: No se encontraron los parámetros para el modelo {model_name}")
        return # Use return instead of continue as it's not in a loop
            
    target_cols = [t.lower() for t in config["targets"]]
    epochs = 1
    patience = 1
    batch_size = params.get("batch_size", 64)

    csv_files = Path(MODEL_TRAIN_DATA / "final_test_filled.tsv") # <--- Ruta corregida
    logger.info(f"Iniciando run de diagnóstico (1 época) con modelo: {model_name}")
    logger.info(f"Cargando archivos desde: {csv_files}")

    print(f"Iniciando run de diagnóstico (1 época) con modelo: {model_name}")
    print(f"Parámetros: {', '.join(f'{k}: {v}' for k, v in params.items())} ")

    # <--- CORRECCIÓN 4: Llamada a train_pipeline (Choice 777) ---
    # La llamada debe coincidir con la definición de la función en pipeline.py
    train_pipeline(
        PGEN_MODEL_DIR,
        csv_files,
        model_name,
        target_cols=target_cols,
        patience = 1,
        epochs = 1
    )
    print(
        "\nRun de diagnóstico completado. Revisa 'comprobacion.txt' o los logs para ver los losses individuales."
    )


    ...

def main():
    
    model_options = list(MODEL_REGISTRY.keys())
    model_name = select_model(model_options, "Selecciona el modelo con el que vas a trabajar: ")
    cli_menu(model_name)

if __name__ == "__main__":
    main()
