""" 
Archivo de configuraciones del modelo: define los modelos disponibles, sus targets, etc.
"""
MODEL_CONFIGS = {
    # ["Drug", "Gene", "Allele", "Genotype", "Outcome_category", "Effect_direction", "Effect_category", "Entity", "Entity_name", "Affected_Pop", "Therapeutic_Outcome"]

    "Outcome-Effect... --> Therapeutic_Outcome": {
        "targets": ["Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"],
        "cols": ["Drugs", "Gene", "Alleles", "Genotype", "Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"]

    },
    "Effect-Entity": {
        "targets": ["Effect", "Entity"],
        "cols": ['Drugs', 'Gene','Alleles', 'Genotype', 'Effect', 'Entity']
    },
    "Variation-Effect": {
        "targets": ["Variation", "Effect"],
        "cols": ['Drugs', 'Gene','Alleles', 'Genotype', 'Variation', 'Effect']
    },
    "Variation-Effect-Entity": {
        "targets": ["Variation", "Effect", "Entity"],
        "cols": ['Drugs', 'Gene','Alleles', 'Genotype', 'Variation', 'Effect', 'Entity']
    }
}

MULTI_LABEL_COLUMN_NAMES = {'outcome', 'effect_subcat', 'entity_affected'}

MASTER_WEIGHTS = {
        "outcome": 1.0,
        "effect_direction": 1.0,
        "effect": 1.0,
        "effect_subcat": 1.5,
        "population_affected": 0.2,
        "entity_affected": 1.5,
        "therapeutic_outcome": 1.0
    }

# 1. Define los hiperparámetros POR DEFECTO
DEFAULT_HYPERPARAMS = {
    "BATCH_SIZE": 64,
    "EPOCHS": 100,
    "LEARNING_RATE": 1e-4, # Un default seguro
    "EMBEDDING_DIM": 256,
    "HIDDEN_DIM": 512,
    "DROPOUT_RATE": 0.3,
    "PATIENCE": 10,
    "WEIGHT_DECAY": 1e-5
}

# 2. Define el REGISTRO de modelos. Cada modelo incluye
#    sus targets, cols, y los HPs que SOBRESCRIBEN el default.
MODEL_REGISTRY = {
    
    "Outcome-Effect... --> Therapeutic_Outcome": {
        "targets": ["Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"],
        "cols": ["Drugs", "Gene", "Alleles", "Genotype", "Outcome", "Effect_direction", "Effect", "Effect_subcat", "Population_Affected", "Entity_Affected", "Therapeutic_Outcome"],
        "params": { # <-- Hiperparámetros específicos
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 2.996788185777191e-05,
            "EMBEDDING_DIM": 128, 
            "HIDDEN_DIM": 704,
            "DROPOUT_RATE": 0.2893325389845274
        }
    },

    "Outcome-Variation": {
        "targets": ["Variation", "Effect"], # Asumo que tienes esto definido
        "cols": ["Drugs", "Gene", "Alleles", "Genotype", "Variation", "Effect"], # Asumo
        "params": {
            "LEARNING_RATE": 0.000177,
            "EMBEDDING_DIM": 512,
            "HIDDEN_DIM": 640,
            "DROPOUT_RATE": 0.13689551662757543
            # Nota: BATCH_SIZE (64), EPOCHS (100) y PATIENCE (10) se heredan del DEFAULT
        }
    },

    "Effect-Entity": {
        "targets": ["Effect", "Entity"],
        "cols": ['Drugs', 'Gene','Alleles', 'Genotype', 'Effect', 'Entity'],
        "params": {
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 2.996788185777191e-05,
            "EMBEDDING_DIM": 64,
            "HIDDEN_DIM": 704,
            "DROPOUT_RATE": 0.2893325389845274
        }
    },
    
    # ... puedes añadir más modelos aquí ...
}

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
        "hyperparams": final_hyperparams  # Un diccionario anidado para HPs
    }
    
    return final_config