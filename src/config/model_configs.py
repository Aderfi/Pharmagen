import sys

from torch._functorch._aot_autograd.logging_utils import model_name
from pathlib import Path
from typing import Dict, Any, Optional, List

#from . import MODEL_REGISTRY, DEFAULT_HYPERPARAMS

def select_model(prompt: str = "Selecciona el modelo:") -> str:
    """
    Muestra los modelos disponibles dentro de la sección [MODEL_REGISTRY]
    y devuelve el nombre de la clave seleccionada.
    """
    full_config = read_toml_config()
    
    # Accedemos al diccionario anidado 'MODEL_REGISTRY'
    registry = full_config.get("MODEL_REGISTRY", {})
    
    if not registry:
        print("Error: No se encontró la sección [MODEL_REGISTRY] en el config.toml")
        sys.exit(1)

    # Obtenemos las claves (ej: "Features-Phenotype", "Phenotype_Effect_Outcome")
    model_options = list(registry.keys())

    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(model_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    
    while True:
        choice = input(f"{prompt} (1-{len(model_options)}): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(model_options):
                return model_options[idx]
        print("Opción no válida. Intente de nuevo.")


'''
def get_model_config(model_name: str, optuna: bool = False) -> dict:
    """
    Obtiene la configuración completa (features, targets e hiperparámetros)
    para el {model_name} especificado.

    Arg:
        model_name (str): Nombre del modelo registrado en MODEL_REGISTRY.

    Returns:
        dict: Configuración completa del modelo.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo desconocido: {model_name}")
    model_conf = MODEL_REGISTRY[model_name]

    final_config = DEFAULT_HYPERPARAMS.copy()
    # print("¡¡¡ SE ESTAN USANDO LOS PARÁMETROS POR DEFECTO !!!!")
    # Si llamamos desde optuna (Pasando optuna=True)
    # usamos los intervalos de búsqueda en lugar de valores fijos.

    if optuna and "params_optuna" in model_conf:
        final_hyperparams = model_conf["params_optuna"]

    final_config = {
        "features": model_conf["features"],
        "targets": model_conf["targets"],
        "stratify": model_conf["stratify_col"],  # Estratificamos por el primer target
        "cols": model_conf["cols"],
        "params": model_conf["params"],
        "weights": model_conf.get("weights"),
    }
    if optuna and "params_optuna" in model_conf:
        final_config["params"] = model_conf["params_optuna"]

    return final_config  # --> dict
'''

def read_toml_config() -> Dict[str, Any]:
    """
    Lee el archivo TOML y devuelve el diccionario crudo.
    """
    import tomli
    toml_file = Path(__file__).resolve().parent / "config.toml"
    
    if not toml_file.exists():
        raise FileNotFoundError(f"El archivo de configuración no existe en: {toml_file}")
        
    with open(toml_file, "rb") as f_conf:
        return tomli.load(f_conf)

def config_parser(*keys_to_extract, model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Obtiene la configuración combinando Defaults + Modelo Específico.

    Args:
        *keys_to_extract: (Opcional) Si se pasan argumentos, solo devuelve esas claves.
                          Si no se pasan, devuelve todo el diccionario del modelo.
        model_name: El nombre exacto del modelo dentro de MODEL_REGISTRY.

    Returns:
        Dict con la configuración final.
    """
    raw_data = read_toml_config()
    
    # 1. Cargar Defaults
    final_config = raw_data.get("DEFAULT_HYPERPARAMS", {}).copy()
    
    # 2. Si hay modelo, fusionar sus datos sobre los defaults
    if model_name:
        registry = raw_data.get("MODEL_REGISTRY", {})
        if model_name in registry:
            model_data = registry[model_name]
            
            # A) Primero copiamos las claves directas del modelo (ej: targets, cols)
            final_config.update(model_data)
            
            # B) Si el modelo tiene una subsección 'params', también la fusionamos al nivel raíz
            #    para que 'learning_rate' del modelo sobrescriba al 'learning_rate' default.
            if "params" in model_data:
                final_config.update(model_data["params"])
        else:
            print(f"Advertencia: El modelo '{model_name}' no existe en el registro.")

    # 3. Filtrado (Si el usuario pidió claves específicas)
    if keys_to_extract:
        filtered_data = {}
        for key in keys_to_extract:
            # Buscamos la clave en nuestra config ya fusionada
            if key in final_config:
                filtered_data[key] = final_config[key]
            else:
                # Opcional: Buscar recursivamente si no está en el primer nivel
                # (Solo si es estrictamente necesario mantener compatibilidad legacy)
                val = _find_key_recursive(raw_data, key) 
                filtered_data[key] = val
        return filtered_data
    
    # 4. Si no pidió claves específicas, devolver TODO el diccionario fusionado
    return final_config

def _find_key_recursive(data: Dict, target_key: str) -> Any:
    """Función auxiliar privada para búsqueda profunda de respaldo."""
    if target_key in data:
        return data[target_key]
    for value in data.values():
        if isinstance(value, dict):
            result = _find_key_recursive(value, target_key)
            if result is not None: return result
    return None



# +----+-------------------------------------+---------+
# |    | Column Names                        |   Index |
# |----+-------------------------------------+---------|
# |  0 | ATC                                 |       0 |
# |  1 | Drug                                |       1 |
# |  2 | Variant/Haplotypes                  |       2 |
# |  3 | Gene                                |       3 |
# |  4 | Alleles                             |       4 |
# |  5 | Phenotype_outcome                   |       5 |
# |  6 | Effect_direction                    |       6 |
# |  7 | Effect_type                         |       7 |
# |  8 | Effect_phenotype                    |       8 |
# |  9 | Effect_phenotype_id                 |       9 |
# | 10 | Metabolizer types                   |      10 |
# | 11 | Population types                    |      11 |
# | 12 | Pop_Phenotypes/Diseases             |      12 |
# | 13 | Comparison Allele(s) or Genotype(s) |      13 |
# | 14 | Comparison Metabolizer types        |      14 |
# | 15 | Notes                               |      15 |
# | 16 | Sentence                            |      16 |
# | 17 | Variant Annotation ID               |      17 |
# +----+-------------------------------------+---------+

"""
DEFAULT_HYPERPARAMS = {
    "embedding_dim": 128,
    "n_layers": 2,
    "hidden_dim": 2048,
    "dropout_rate": 0.2,
    "weight_decay": 1e-3,
    "learning_rate": 1e-4,
    "batch_size": 128,
}


MULTI_LABEL_COLUMN_NAMES = {  # Columnas que son MULTI-LABEL
    "phenotype_outcome",
}

MODEL_REGISTRY = {
    "Phenotype_Effect_Outcome": {  # choice 1
        "targets": ["Phenotype_outcome", "Effect_direction", "Effect_type"],
        "inputs": ["Drug", "Genalle", "Gene", "Allele"],
        "cols": [
            "Drug",
            "Genalle",
            "Gene",
            "Allele",
            "Phenotype_outcome",
            "Effect_direction",
            "Effect_type",
        ],
        # "cols": [
        #    "ATC",
        #    "Drug",
        #    "Genalle",
        #    #"Variant/Haplotypes",
        #    "Gene",
        #    "Allele",
        #    "Phenotype_outcome",
        #    "Effect_direction",
        #    "Effect_type",
        #    "Effect_phenotype",
        # ],
        "params": {
            "embedding_dim": 256,
            "n_layers": 2,
            "hidden_dim": 2176,
            "dropout_rate": 0.6826665445146843,
            "weight_decay": 0.0001478755319922093,
            "learning_rate": 0.0002023597603520488,
            "batch_size": 64,
            "attention_dim_feedforward": 3840,
            "attention_dropout": 0.48387265129374835,
            "num_attention_layers": 2,
            "focal_gamma": 1.1605241620974445,
            "focal_alpha_weight": 1.3417203741569734,
            "label_smoothing": 0.11100845590336689,
            "optimizer_type": "rmsprop",
            "manual_task_weights": True,
            "use_batch_norm": True,
            "use_layer_norm": True,
            "activation_function": "gelu",
            "gradient_clip_norm": 2.231503593586354,
            "fm_dropout": 0.34731215338736454,
            "fm_hidden_layers": 2,
            "fm_hidden_dim": 384,
            "embedding_dropout": 0.1546248948072961,
            "early_stopping_patience": 25,
            "scheduler_type": "none",
        },
        "params_optuna": {
            # === CORE ARCHITECTURE ===
            "embedding_dim": [128, 256, 384, 512, 640, 768],
            "n_layers": ["int", 2, 8, 1],
            "hidden_dim": ["int", 512, 4096, 128],
            "activation_function": ["gelu", "relu", "swish", "mish"],
            # === REGULARIZATION ===
            "dropout_rate": (0.1, 0.7),
            "weight_decay": (1e-6, 1e-2),
            "label_smoothing": (0.0, 0.3),
            "embedding_dropout": (0.1, 0.4),
            "gradient_clip_norm": (0.5, 5.0),
            # === BATCH & LAYER NORMALIZATION ===
            "use_batch_norm": [True, False],
            "use_layer_norm": [True, False],
            # === OPTIMIZER & LEARNING RATE ===
            "learning_rate": (1e-5, 1e-3),
            "batch_size": [64, 128, 256],
            "optimizer_type": ["adamw", "adam", "rmsprop"],
            "adam_beta1": (0.85, 0.95),
            "adam_beta2": (0.95, 0.999),
            # === SCHEDULER ===
            "scheduler_type": ["plateau", "cosine", "exponential", "none"],
            "scheduler_factor": (0.1, 0.7),
            "scheduler_patience": ["int", 3, 10, 1],
            "early_stopping_patience": [
                "int",
                15,
                40,
                5,
            ],  # Optuna will test values from 15 to 40
            # === ATTENTION MECHANISM ===
            "num_attention_layers": [1, 2, 3, 4],
            "attention_dim_feedforward": ["int", 512, 4096, 256],
            "attention_dropout": (0.1, 0.5),
            # === FOCAL LOSS (for effect_type task) ===
            "focal_gamma": (1.0, 5.0),
            "focal_alpha_weight": (0.25, 4.0),
            # === FM (FACTORIZATION MACHINE) BRANCH ===
            "fm_dropout": (0.1, 0.5),
            "fm_hidden_layers": [0, 1, 2],
            "fm_hidden_dim": ["int", 64, 512, 64],
            # === TASK WEIGHTING STRATEGY ===
            # Let Optuna decide between fixed priorities or learned weights
            "manual_task_weights": [True, False],
        },
    },
    "Features-Phenotype": {  # choice 2
        "stratify_col": ["Phenotype_outcome"],
        "targets": ["Phenotype_outcome"],
        "features": ["atc", "Genalle", "Gene", "Allele"],
        "cols": [
            "Drug",
            "Genalle",
            "Gene",
            "Allele",
            "Phenotype_outcome",
            "Effect_direction",
            "Effect_type",
        ],
        "params": {
            "batch_size": 512,
            "embedding_dim": 192,
            "n_layers": 1,
            "hidden_dim": 1024,
            "dropout_rate": 0.6996731804616522,
            "learning_rate": 0.0001271487593514609,
            "weight_decay": 0.001761363247217078,
        },
        "params_optuna": {
            # === CORE ARCHITECTURE ===
            "embedding_dim": [128, 256, 512],
            "n_layers": ["int", 2, 4, 1],
            "hidden_dim": ["int", 512, 4096, 128],
            "activation_function": ["gelu", "relu"],  # "swish", "mish"],
            # === REGULARIZATION ===
            "dropout_rate": (0.1, 0.5),
            "weight_decay": (1e-6, 1e-2),
            "label_smoothing": (0.0, 0.3),
            "embedding_dropout": (0.1, 0.4),
            "gradient_clip_norm": (0.5, 5.0),
            # === BATCH & LAYER NORMALIZATION ===
            "use_batch_norm": [True, False],
            "use_layer_norm": [True, False],
            # === OPTIMIZER & LEARNING RATE ===
            "learning_rate": (1e-5, 1e-3),
            "batch_size": [64, 128, 256],
            "optimizer_type": ["adamw", "adam", "rmsprop"],
            "adam_beta1": (0.85, 0.95),
            "adam_beta2": (0.95, 0.999),
            # === SCHEDULER ===
            "scheduler_type": ["plateau", "cosine", "exponential", "none"],
            "scheduler_factor": (0.1, 0.7),
            "scheduler_patience": ["int", 3, 10, 1],
            "early_stopping_patience": [
                "int",
                15,
                40,
                5,
            ],  # Optuna will test values from 15 to 40
            # === ATTENTION MECHANISM ===
            "num_attention_layers": [1, 2, 3, 4],
            "attention_dim_feedforward": ["int", 512, 4096, 256],
            "attention_dropout": (0.1, 0.5),
            # === FOCAL LOSS (for effect_type task) ===
            "focal_gamma": (1.0, 5.0),
            "focal_alpha_weight": (0.25, 4.0),
            # === FM (FACTORIZATION MACHINE) BRANCH ===
            "fm_dropout": (0.1, 0.5),
            "fm_hidden_layers": [0, 1, 2],
            "fm_hidden_dim": ["int", 64, 512, 64],
            # === TASK WEIGHTING STRATEGY ===
            # Let Optuna decide between fixed priorities or learned weights
            "manual_task_weights": [True, False],
        },
        "weights": {
            "phenotype_outcome": 1.0,
            "effect_direction": 1.0,
            "effect_type": 1.0,
            "effect_phenotype": 1.0,
        },
    },
    "Features_PhenEff": {},
}


if __name__ == "__main__":  # Prueba rápida de la función get_model_config
    model_options = list(MODEL_REGISTRY.keys())
    modelo = select_model(model_options=model_options)
    config = get_model_config(modelo)
    print(f"Configuración para el modelo '{modelo}':")

    print(config.keys())

    print("\nTargets:", config["targets"])
    print("\nColumns:", config["cols"])
    print("\nParameters:", config["params"])

    params = config["params"]

    print("\nLearning Rate:", params["learning_rate"])
    print("Batch Size:", params["batch_size"])
"""

if __name__ == "__main__":
    
    #items = ["Features-Phenotype"]
    #final_config = config_parser(*items)

    #print("Configuración final en memoria:")

    #print(len(final_config))

    model_choice = select_model()

    model_config = config_parser(model_name=model_choice)

    print(len(model_config))