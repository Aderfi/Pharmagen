import os
from keras.models import load_model
from Pharmagen.src.data_handle.preprocess_model import PharmagenPreprocessor

def load_pharmagen_model(model_path, mut_vocab_path, drug_vocab_path):
    """
    Carga el modelo Keras entrenado y los vocabularios de mutaciones y medicamentos.
    
    Args:
        model_path (str): Ruta al archivo .h5 del modelo entrenado.
        mut_vocab_path (str): Ruta al JSON del vocabulario de mutaciones.
        drug_vocab_path (str): Ruta al JSON del vocabulario de medicamentos.
    
    Returns:
        model: El modelo Keras cargado.
        preproc: Un objeto PharmagenPreprocessor con los vocabularios cargados.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")
    model = load_model(model_path)
    
    preproc = PharmagenPreprocessor()
    preproc.load_vocabs(mut_vocab_path, drug_vocab_path)
    return model, preproc

def predict_with_pharmagen(model, preproc, df, mut_col="mutaciones", drug_col="medicamentos"):
    """
    Preprocesa un DataFrame de entrada y obtiene predicciones del modelo.
    
    Args:
        model: El modelo Keras cargado.
        preproc: Objeto PharmagenPreprocessor con vocabularios cargados.
        df (pd.DataFrame): DataFrame con las columnas de mutaciones y medicamentos.
        mut_col (str): Nombre de la columna de mutaciones.
        drug_col (str): Nombre de la columna de medicamentos.
    
    Returns:
        np.ndarray: Vector de predicciones.
    """
    X_mut, X_drug = preproc.transform(df, mut_col, drug_col)
    preds = model.predict([X_mut, X_drug])
    return preds