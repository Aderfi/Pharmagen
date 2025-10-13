# ======== AJUSTE DE PARÁMETROS ===========
# Hiperparámetros óptimos para cada modelo.
# Se definen aquí para facilitar su modificación y uso en los scripts de entrenamiento.
# Han sido encontrados mediante Optuna y validación cruzada.
# ========================================

def metrics_models(model_name):
    # Normaliza el nombre a mayúsculas y quita espacios
    norm_name = model_name.replace(" ", "").upper()
    if norm_name == "OUTCOME-VARIATION-EFFECT-ENTITY":
        # Modelo Outcome-Variation-Effect-Entity
        BATCH_SIZE = 8
        EPOCHS = 100
        LEARNING_RATE = 2.996788185777191e-05
        EMB_DIM = 64
        HIDDEN_DIM = 704
        DROPOUT_RATE = 0.2893325389845274
        PATIENCE = 10
        return BATCH_SIZE, EPOCHS, LEARNING_RATE, EMB_DIM, HIDDEN_DIM, DROPOUT_RATE, PATIENCE
    
    elif norm_name == "OUTCOME-VARIATION":
        # Modelo: OUTCOME - VARIATION
        BATCH_SIZE = 8
        EPOCHS = 100
        LEARNING_RATE = 2.996788185777191e-05
        EMB_DIM = 64
        HIDDEN_DIM = 704
        DROPOUT_RATE = 0.2893325389845274
        PATIENCE = 10
        return BATCH_SIZE, EPOCHS, LEARNING_RATE, EMB_DIM, HIDDEN_DIM, DROPOUT_RATE, PATIENCE
    
    elif norm_name == "EFFECT-ENTITY":
        # Modelo: EFFECT - ENTITY
        BATCH_SIZE = 8
        EPOCHS = 100
        LEARNING_RATE = 2.996788185777191e-05
        EMB_DIM = 64
        HIDDEN_DIM = 704
        DROPOUT_RATE = 0.2893325389845274
        PATIENCE = 10
        return BATCH_SIZE, EPOCHS, LEARNING_RATE, EMB_DIM, HIDDEN_DIM, DROPOUT_RATE, PATIENCE
    
    else:
        raise ValueError(f"Modelo '{model_name}' no reconocido")
    
'''
# ======== Rutas y archivos ===========
# Rutas y archivos para el modelo.
# Se definen aquí para facilitar su modificación y uso.
# ========================================
'''

DATA_PATH = "train_data"
MODEL_PATH_EXT = "models/{}.pth"
SAVE_ENCODERS_AS = "encoders{}.pkl"
RESULTS_DIR = "../results/"

