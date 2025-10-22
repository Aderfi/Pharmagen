""" 
Archivo de configuraciones del modelo: define los modelos disponibles, sus targets, etc.
"""
def get_model_config(model_name: str) -> dict:
    """
    Obtiene la configuración completa (targets, cols e hiperparámetros)
    para un modelo dado, fusionando defaults y específicos.
    """
    # 1. Normaliza el nombre (o busca por el nombre exacto)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Modelo '{model_name}' no reconocido. Modelos disponibles: {list(MODEL_REGISTRY.keys())}")
    
    # 2. Coge la configuración específica del modelo
    model_conf = MODEL_REGISTRY[model_name]
    
    # 3. Empieza con los HPs por defecto
    final_hyperparams = DEFAULT_HYPERPARAMS.copy()
    
    # 4. Sobrescribe con los HPs específicos del modelo
    if "params" in model_conf:
        final_hyperparams.update(model_conf["params"])
        
    # 5. Devuelve todo en un solo diccionario, limpio y seguro
    final_config = {
        "targets": model_conf["targets"],
        "cols": model_conf["cols"],
        "hyperparams": final_hyperparams,
        "weights": model_conf.get("weights")
        # Un diccionario anidado para HPs
    }
    
    return final_config

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

DEFAULT_HYPERPARAMS = {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 1e-4, # Un default seguro
    "embedding_dim": 256,
    "hidden_dim": 512,
    "dropout_rate": 0.3,
    "patience": 10,
    "weight_decay": 1e-5
}

MASTER_WEIGHTS = {
    "outcome": 1.0,
    "effect_direction": 1.0,  
    "effect": 1.0,             
    "effect_subcat": 1.0,        
    "entity_affected": 1.0      
    #"population_affected": 1.0, 
    #"therapeutic_outcome": 1.0  # 
}

MULTI_LABEL_COLUMN_NAMES = {    # Columnas que son MULTI-LABEL
                                'outcome', 
                                #'effect',
                                'effect_direction',
                                'effect_subcat' 
                                #'entity_affected'
}    # Columnas que son MULTI-LABEL

MODEL_REGISTRY = {
    "Outcome-Effect... --> Therapeutic_Outcome": {
        "targets": ["Outcome", "Effect_direction", "Effect", "Effect_subcat", "Entity_Affected", "Population_Affected", "Therapeutic_Outcome"],
        "cols": ["Drug", "Gene", "Allele", "Genotype", "Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"],
        "params": { # <-- Hiperparámetros específicos
            "batch_size": 64,
            "embedding_dim": 512,
            "hidden_dim": 1536,
            "dropout_rate": 0.25,
            "learning_rate": 0.000944,
            "weight_decay": 7.24e-05
        }
    },

    "Outcome-Effect-Subcat-Entity": {"targets": ["Outcome", "Effect_direction", "Effect", "Effect_subcat", "Entity_Affected"],
                                     
                                     "cols":    ["Drug", "Gene", "Allele", "Genotype", "Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"],
                                     
                                     "params":  {"batch_size": 64, "embedding_dim": 512,"hidden_dim": 1536,
                                                 "dropout_rate": 0.25, "learning_rate": 0.000944, 
                                                 "weight_decay": 7.24e-05}
                                    },

    "Outcome-Effect-Subcat":   {"targets": ["Outcome", "Effect_direction", "Effect", "Effect_subcat"],
                              
                                "cols": ["Drug", "Gene", "Allele", "Genotype", "Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"],
                                
                                "params": {"batch_size": 64, "embedding_dim": 768, "hidden_dim": 1024,
                                             "dropout_rate": 0.4, "learning_rate": 0.000782,
                                             "weight_decay": 1.31e-06},

                                "weights": {"outcome": 1.0, "effect_direction": 0.45, "effect": 10.0, "effect_subcat": 40.0}
    },
}