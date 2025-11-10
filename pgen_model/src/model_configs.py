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
    #path_abs = Path(MODELS_DIR / carpeta_guardado)

    # 5. Devuelve todo en un solo diccionario, limpio y seguro
    
    final_config = {
            "targets": model_conf["targets"],
            "cols": model_conf["cols"],
            "params": final_hyperparams,
            "weights": model_conf.get("weights")
            # Un diccionario anidado para HPs
        }
    
    
    return final_config  # --> dict

#+----+-------------------------------------+---------+
#|    | Column Names                        |   Index |
#|----+-------------------------------------+---------|
#|  0 | ATC                                 |       0 |
#|  1 | Drug                                |       1 |
#|  2 | Variant/Haplotypes                  |       2 |
#|  3 | Gene                                |       3 |
#|  4 | Alleles                             |       4 |
#|  5 | Phenotype_outcome                   |       5 |
#|  6 | Effect_direction                    |       6 |
#|  7 | Effect_type                         |       7 |
#|  8 | Effect_phenotype                    |       8 |
#|  9 | Effect_phenotype_id                 |       9 |
#| 10 | Metabolizer types                   |      10 |
#| 11 | Population types                    |      11 |
#| 12 | Pop_Phenotypes/Diseases             |      12 |
#| 13 | Comparison Allele(s) or Genotype(s) |      13 |
#| 14 | Comparison Metabolizer types        |      14 |
#| 15 | Notes                               |      15 |
#| 16 | Sentence                            |      16 |
#| 17 | Variant Annotation ID               |      17 |
#+----+-------------------------------------+---------+


CLINICAL_PRIORITIES = {
    'effect_type': 0.6,           # 50% - Crítico (toxicidad/eficacia)
    'phenotype_outcome': 0.25,     # 30% - Importante (resultado clínico)
    'effect_direction': 0.15,      # 20% - Útil (dirección del efecto)
}

DEFAULT_HYPERPARAMS = {
        "embedding_dim": [128, 256, 384, 512, 640, 768, 1024],
        "n_layers": [2, 3, 4, 5, 6, 7],
        "hidden_dim": ["int", 512, 4096, 256],
            
        "dropout_rate": (0.1, 0.7),
        "weight_decay": (1e-6, 1e-2),
            
        "learning_rate": (1e-5, 1e-3),
        "batch_size": [64, 128, 256, 512],
            
            # === TRANSFORMER ATTENTION (based on your attention_layer) ===
            "attention_dim_feedforward": ["int", 512, 4096, 256],  # For transformer feedforward
            "attention_dropout": (0.1, 0.5),
            "num_attention_layers": [1, 2, 3, 4],  # Stack multiple transformer layers
            
            # === FOCAL LOSS (you're using it for effect_type) ===
            "focal_gamma": (1.0, 5.0),  # Currently hardcoded to 2.0
            "focal_alpha_weight": (0.5, 3.0),  # Multiplier for class weights
            "label_smoothing": (0.05, 0.3),  # Currently hardcoded to 0.15
            
            # === OPTIMIZER VARIANTS ===
            "optimizer_type": ["adamw", "adam", "sgd", "rmsprop"],
            "adam_beta1": (0.8, 0.95),
            "adam_beta2": (0.95, 0.999),
            "sgd_momentum": (0.85, 0.99),
            
            # === SCHEDULER ===
            "scheduler_type": ["plateau", "cosine", "step", "exponential", "none"],
            "scheduler_factor": (0.1, 0.8),  # For ReduceLROnPlateau
            "scheduler_patience": ["int", 3, 15, 1],
            
            # === TASK WEIGHTING ===
            "uncertainty_weighting": [True, False],  # Toggle your uncertainty weighting
            "manual_task_weights": [True, False],  # Use CLINICAL_PRIORITIES vs learned weights
            
            # === ADVANCED ARCHITECTURE ===
            "use_batch_norm": [True, False],
            "use_layer_norm": [True, False], 
            "activation_function": ["gelu", "relu", "swish", "mish"],  # You're using GELU
            "gradient_clip_norm": (0.5, 5.0),
            
            # === FM BRANCH ENHANCEMENTS ===
            "fm_dropout": (0.1, 0.5),  # Separate dropout for FM interactions
            "fm_hidden_layers": [0, 1, 2],  # Add layers after FM interactions
            "fm_hidden_dim": ["int", 64, 512, 32],
            
            # === EMBEDDING VARIATIONS ===
            "embedding_dropout": (0.1, 0.3),
            "drug_embedding_dim": ["int", 64, 1024, 32],  # Separate embedding dims
            "gene_embedding_dim": ["int", 64, 1024, 32],
            "allele_embedding_dim": ["int", 32, 512, 16],
            "genalle_embedding_dim": ["int", 64, 1024, 32],
            
            # === TRAINING DYNAMICS ===
            "warmup_epochs": ["int", 0, 20, 1],
            "early_stopping_patience": ["int", 15, 50, 5],
            "validation_frequency": ["int", 1, 5, 1],  # Validate every N epochs
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
    "phenotype_outcome",
    #"effect_phenotype",
    #"pop_phenotypes/diseases",
} 

MODEL_REGISTRY = {
    "Phenotype_Effect_Outcome": {   # choice 1
        "targets": ["Phenotype_outcome", "Effect_direction", "Effect_type"],
        "cols": [
            "ATC",
            "Drug",
            "Genalle",
            #"Variant/Haplotypes",
            "Gene",
            "Allele",
            "Phenotype_outcome",
            "Effect_direction",
            "Effect_type",
            "Effect_phenotype",
        ],
        "params": {
            "batch_size": 64,
            "embedding_dim": 384,
            "n_layers": 4,
            "hidden_dim": 1280,
            "dropout_rate": 0.391,
            "learning_rate": 6.32e-05,
            "weight_decay": 4.30e-05
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
            "optimizer_type": ["adamw", "adam", "rmsprop"], # SGD can be slow to converge
            "adam_beta1": (0.85, 0.95),
            "adam_beta2": (0.95, 0.999),
            
            # === SCHEDULER ===
            "scheduler_type": ["plateau", "cosine", "exponential", "none"],
            "scheduler_factor": (0.1, 0.7),
            "scheduler_patience": ["int", 3, 10, 1],
            "early_stopping_patience": ["int", 15, 40, 5], # Optuna will test values from 15 to 40

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
    "ATC_Phenotype_Effect_Outcome": {   # choice 2
        "targets": ["Phenotype_outcome", "Effect_direction", "Effect_type", "Effect_phenotype", "Effect_phenotype_id", "Pop_Phenotypes/Diseases", "Pop_phenotype_id"],
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
            "Effect_phenotype_id",
            "Pop_Phenotypes/Diseases",
            "Pop_phenotype_id",
            "Metabolizer types",
            "Population types",
            "Comparison Allele(s) or Genotype(s)",
            "Comparison Metabolizer types",
            "Notes",
            "Sentence",
            "Variant Annotation ID"
        ],

        "params": {
            "batch_size": 512,
            "embedding_dim": 192,
            "n_layers": 1,
            "hidden_dim": 1024,
            "dropout_rate": 0.6996731804616522,
            "learning_rate": 0.0001271487593514609,
            "weight_decay": 0.001761363247217078
            },
        
        "params_optuna": {
            "batch_size": [512],
            "embedding_dim": [192],
            "n_layers": [1],
            "hidden_dim": ["int", 512, 1024, 256],
            "dropout_rate": (0.4, 0.7),
            "learning_rate": (1e-4, 6e-4),
            "weight_decay": (1e-3, 1e-2)
        },

        "weights": {
            "phenotype_outcome": 1.0,
            "effect_direction": 1.0,
            "effect_type": 1.0,
            "effect_phenotype": 1.0,
            
            
            
            
        },
    },
}


if __name__ == "__main__":    # Prueba rápida de la función get_model_config
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
    