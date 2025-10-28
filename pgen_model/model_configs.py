"""
Archivo de configuraciones del modelo: define los modelos disponibles, sus targets, etc.
"""
from pathlib import Path
from src.config.config import *

def select_model(model_options, prompt="Selecciona el modelo:"):
    """
    Muestra una lista numerada de modelos y pide al usuario que seleccione uno.
    Devuelve el nombre (string) del modelo seleccionado. {model_name}
    """

    print("\n————————————————— Modelos Disponibles ————————————————")
    for i, name in enumerate(model_options, 1):
        print(f"  {i} -- {name}")
    print("———————————————————————————————————————————————————————")
    model_choice = ""
    while model_choice not in [str(i + 1) for i in range(len(model_options))]:
        model_choice = input(f"{prompt} (1-{len(model_options)}): ").strip()
        if model_choice not in [str(i + 1) for i in range(len(model_options))]:
            print("Opción no válida. Intente de nuevo.")
    return model_options[int(model_choice) - 1]


def get_model_config(model_name: str, path=False) -> dict:
    """
    Obtiene la configuración completa (targets, cols e hiperparámetros)
    para el {model_name} especificado.
    """
    # 1. Normaliza el nombre (o busca por el nombre exacto)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' no reconocido. Modelos disponibles: {list(MODEL_REGISTRY.keys())}"
        )

    # 2. Coge la configuración específica del modelo
    model_conf = MODEL_REGISTRY[model_name]

    # 3. Empieza con los HPs por defecto
    final_hyperparams = DEFAULT_HYPERPARAMS.copy()

    # 4. Sobrescribe con los HPs específicos del modelo
    if "params" in model_conf:
        final_hyperparams.update(model_conf["params"])


    carpeta_guardado = str(model_conf.get("path"))
    path_abs = Path(MODELS_DIR / carpeta_guardado)

    # 5. Devuelve todo en un solo diccionario, limpio y seguro
    '''
    if path == True:
        final_config = {
            "path": str(path_abs),
        }
    else:
    '''
    final_config = {
            "targets": model_conf["targets"],
            "cols": model_conf["cols"],
            "params": final_hyperparams,
            "weights": model_conf.get("weights")
            #"path": str(path_abs),
            # Un diccionario anidado para HPs
        }
    
    
    return final_config  # --> dict


DEFAULT_HYPERPARAMS = {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-4,  # Un default seguro
    "embedding_dim": 256,
    "hidden_dim": 512,
    "dropout_rate": 0.3,
    "patience": 10,
    "weight_decay": 1e-5,
}

MASTER_WEIGHTS = {
    "outcome": 1.0,
    "effect_direction": 1.0,
    "effect": 1.0,
    "effect_subcat": 1.0,
    "entity_affected": 1.0,
    # "population_affected": 1.0,
    # "therapeutic_outcome": 1.0  #
}

MULTI_LABEL_COLUMN_NAMES = {  # Columnas que son MULTI-LABEL
    "outcome",
    "var_type",
    "name",
    "effect",
    "effect_subcat",
    "entity_affected",
}  # Columnas que son MULTI-LABEL

MODEL_REGISTRY = {
    "Outcome-Effect... --> Therapeutic_Outcome": {  # choice 1 
        "targets": [
            "Outcome",
            "Effect_direction",
            "Effect",
            "Effect_subcat",
            "Entity_Affected",
            "Population_Affected",
            "Therapeutic_Outcome",
        ],
        "cols": [
            "Drug",
            "Gene",
            "Allele",
            "Genotype",
            "Outcome",
            "Effect_direction",
            "Effect",
            "Effect_subcat",
            "Population_Affected",
            "Entity_Affected",
            "Therapeutic_Outcome",
        ],
        "params": {  # <-- Hiperparámetros específicos
            "batch_size": 64,
            "embedding_dim": 512,
            "hidden_dim": 1536,
            "dropout_rate": 0.25,
            "learning_rate": 0.000944,
            "weight_decay": 7.24e-05,
        },
    },
    "Outcome-Effect-Subcat-Entity": {   # choice 2
        "targets": [
            "Outcome",
            "Effect_direction",
            "Effect",
            "Effect_subcat",
            "Entity_Affected",
        ],
        "cols": [
            "Drug",
            "Gene",
            "Allele",
            "Genotype",
            "Outcome",
            "Effect_direction",
            "Effect",
            "Effect_subcat",
            "Population_Affected",
            "Entity_Affected",
            "Therapeutic_Outcome",
        ],
        "params": {
            "batch_size": 64,
            "embedding_dim": 512,
            "hidden_dim": 1536,
            "dropout_rate": 0.25,
            "learning_rate": 0.000944,
            "weight_decay": 7.24e-05,
        },
    },
    "Outcome-Effect-Subcat": {  # choice 3
        "targets": ["Outcome", "Effect", "Effect_subcat"],
        "cols": [
            "Drug",
            "Gene",
            "Allele",
            "Genotype",
            "Outcome",
            "Effect",
            "Effect_subcat",
            "Population_Affected",
            "Entity_Affected",
            "Therapeutic_Outcome",
        ],
        "params": {
            "batch_size": 64,
            "embedding_dim": 1024,
            "hidden_dim": 1024,
            "dropout_rate": 0.34859758458568946,
            "learning_rate": 0.0003851389762174168,
            "weight_decay": 5.002211072772415e-06,
        },
        "params_optuna": {
            "batch_size": [64, 128],
            "embedding_dim": [1024, 2048],
            "hidden_dim": [1024, 2048],
            "dropout_rate": (0.2, 0.4),
            "learning_rate": (1e-4, 1e-3),
            "weight_decay": (1e-7, 1e-5),
        },
        "weights": {
            "outcome": 1.0,  # 0.142,
            # "effect_direction": 1.0, #0.072,
            "effect": 1.0,  # 0.018,
            "effect_subcat": 1.0,  # 0.767
        },
        "path": "Outcome_Effect_Subcat",
    },
    #Drug;Gene;Allele;Genotype;Outcome;Variation;Type;Name;Therapeutic_Outcome

    "Outcome-Variation-Var_Type-Name": {     # choice 4
        "targets": ["Outcome", "Variation", "Var_Type", "Therapeutic_Outcome"],
        "cols": [
            "Drug",
            "Gene",
            "Allele",
            "Genotype",
            "Outcome",
            "Variation",
            "Var_Type",
            "Name",
            "Therapeutic_Outcome",
        ],
        "params": {
            "batch_size": 64,
            "embedding_dim": 512,
            "hidden_dim": 768,
            "dropout_rate": 0.24667,
            "learning_rate": 0.000728,
            "weight_decay": 7.51e-06,
        },
        "params_optuna": {
            # Búsqueda más ajustada
            "batch_size": [32, 64],
            
            # AMPLIADO: El mejor estaba en el borde (768)
            "embedding_dim": ["int", 640, 1024, 128], 
            
            # REDUCIDO: El mejor estaba en el centro (1536)
            "hidden_dim": ["int", 1024, 2048, 256],    
            
            # REDUCIDO: Centrado alrededor de 0.25
            "dropout_rate": (0.2, 0.35),
            
            # REDUCIDO: Centrado alrededor de 3.5e-4
            "learning_rate": (1e-4, 5e-4),
            
            # REDUCIDO: Centrado alrededor de 1e-5
            "weight_decay": (5e-6, 5e-5),
        },
        "weights": {
            "outcome": 1.0,
            "variation": 1.0,
            "type": 1.0,
            "name": 1.0,
        },
    },
    "Subcat-Entity": {   # choice 5
        "targets": ["Effect_subcat", "Population_Affected", "Entity_Affected"],
        "cols": [
            "Drug",
            "Gene",
            "Allele",
            "Genotype",
            "Outcome",
            "Effect_direction",
            "Effect",
            "Effect_subcat",
            "Population_Affected",
            "Entity_Affected",
            "Therapeutic_Outcome",
        ],
        "params": {
            "batch_size": 64,
            "embedding_dim": 768,
            "hidden_dim": 1024,
            "dropout_rate": 0.4,
            "learning_rate": 0.000782,
            "weight_decay": 1.31e-06,
        },
        "params_optuna": {
            "batch_size": [64, 128, 256],
            "embedding_dim": [128, 256, 512],
            "hidden_dim": [256, 512, 1024],
            "dropout_rate": [0.1, 0.2, 0.3],
            "learning_rate": [1e-5, 1e-4, 1e-3],
            "weight_decay": [1e-6, 1e-5, 1e-4],
        },
        "weights": {
            "outcome": 1.0,
            "effect_direction": 1.0,
            "effect": 1.0,
            "effect_subcat": 1.0,
        },
    },
}
