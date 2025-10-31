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


def get_model_config(model_name: str) -> dict:
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
    
    final_config = {
            "targets": model_conf["targets"],
            "cols": model_conf["cols"],
            "params": final_hyperparams,
            "weights": model_conf.get("weights")
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
    "patience": 20,
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
    "phenotype_outcome"
} 

MODEL_REGISTRY = {
    "Phenotype_Effect_Outcome": {   # choice 1
        "targets": ["Phenotype_outcome", "Effect_direction", "Effect_type", "Effect_phenotype"],
        "cols": [
            "ATC",
            "Drug",
            "Variant/Haplotypes",
            "Gene",
            "Allele",
            "Phenotype_outcome",
            "Effect_direction",
            "Effect_type",
            "Effect_phenotype",
            "Metabolizer types",
            "Population types",
            "Pop_Phenotypes/Diseases",
            "Comparison Allele(s) or Genotype(s)",
            "Comparison Metabolizer types",
            "Notes",   
            "Sentence",
            "Variant Annotation ID"
        ],
        
        "params": {
            "batch_size": 512,
            "embedding_dim": 128,
            "hidden_dim": 4096,
            "dropout_rate": 0.493740315869821,
            "learning_rate": 0.0005893232367500335,
            "weight_decay": 3.308304117490965e-05
        },
        "params_optuna": {
            "batch_size": [512],
            "embedding_dim": [128],
            "hidden_dim": [4096],
            "dropout_rate": (0.4, 0.5),
            "learning_rate": (5e-4, 8e-4),
            "weight_decay": (2e-5, 4e-5),
        },
        
        "weights": {
            "phenotype_outcome": 1.0,
            "effect_direction": 1.0,
            "effect_type": 1.0,
            "effect_phenotype": 1.0,
        },
    },
}
