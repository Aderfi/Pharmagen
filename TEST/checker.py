import pandas as pd
import re
import json
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
import joblib

# Asumimos que tu CSV tiene columnas como 'drug_name', 'gene_name', 'genotype'

df = pd.read_csv("train_therapeutic_outcome_cleaned.csv", sep=";", index_col=False, usecols=["Drug", "Gene", "Allele", "Genotype", "Outcome", "Effect", "Effect_subcat"])
        
        
    
def split_items(text):
    """
    Toma un string, lo divide por varios delimitadores 
    (coma, +, /, espacio) y devuelve una lista limpia de items.
    """
    # 1. Comprobar si la entrada es un string. Si es NaN o None, devolver lista vacía.
    if not isinstance(text, str):
        return []
    
    # 2. Definir el patrón regex para dividir
    #    Esto divide por CUALQUIERA de los siguientes:
    #    , (coma), + (más), / (slash), \s (espacio)
    #    El [ ]+ significa "uno o más de estos delimitadores"
    regex_pattern = r'[,+/\s]+'
    
    # 3. Dividir el string usando el regex
    items = re.split(regex_pattern, text)
    
    # 4. Limpiar la lista:
    #    - .strip() elimina espacios en blanco al inicio/final
    #    - 'if item.strip()' elimina strings vacíos ('')
    cleaned_items = [item.strip() for item in items if item.strip()]
    
    return cleaned_items

def split_classes_subcat(text):
    """
    Toma un string, lo divide por varios delimitadores 
    (coma, +, /, espacio) y devuelve una lista limpia de items.
    """
    # 1. Comprobar si la entrada es un string. Si es NaN o None, devolver lista vacía.
    if not isinstance(text, str):
        return []
    
    # 2. Definir el patrón regex para dividir
    #    Esto divide por CUALQUIERA de los siguientes:
    #    , (coma), + (más), / (slash), \s (espacio)
    #    El [ ]+ significa "uno o más de estos delimitadores"
    regex_pattern = r'[,]+'
    regex_next = r'[:]+'
    
    # 3. Dividir el string usando el regex
    items = re.split(regex_pattern, text)
    items_down = re.split(regex_next, items[0])
    
    # 4. Limpiar la lista:
    #    - .strip() elimina espacios en blanco al inicio/final
    #    - 'if item.strip()' elimina strings vacíos ('')
    cleaned_items = [item.strip() for item in items_down if item.strip()]
    
    return cleaned_items




# Aplicamos la función
df['drug_list'] = df['Drug'].apply(split_items)
df['allele_list'] = df['Allele'].apply(split_items)
df['geno_list'] = df['Genotype'].apply(split_items)
df['gene_list'] = df['Gene'].apply(lambda x: x.strip() if isinstance(x, str) else x)

df['outcome_list'] = df['Outcome'].apply(split_classes_subcat)
df['effect_subcat_list'] = df['Effect_subcat'].apply(split_classes_subcat)
df['effect_list'] = df['Effect'].apply(split_classes_subcat)



# --- 3. Crear y Entrenar los Encoders ---
mlb_drug = MultiLabelBinarizer()
mlb_geno = MultiLabelBinarizer()
mlb_allele = MultiLabelBinarizer()
label_gene = LabelEncoder()


mlb_outcome = MultiLabelBinarizer()
mlb_effect_subcat = MultiLabelBinarizer()
mlb_effect = MultiLabelBinarizer()

# Entrenamos (fit) CADA UNO en su columna de listas
mlb_drug.fit(df['drug_list'])
mlb_geno.fit(df['geno_list'])
mlb_allele.fit(df['allele_list'])
label_gene.fit(df['gene_list'])

mlb_outcome.fit(df['outcome_list'])
mlb_effect_subcat.fit(df['effect_subcat_list'])
mlb_effect.fit(df['effect_list'])

# --- 4. Guardar los Encoders (¡CRUCIAL!) ---
joblib.dump(mlb_drug, 'mlb_drug_encoder.joblib')
joblib.dump(mlb_geno, 'mlb_geno_encoder.joblib')
joblib.dump(mlb_allele, 'mlb_allele_encoder.joblib')
joblib.dump(label_gene, 'label_gene_encoder.joblib')
joblib.dump(mlb_outcome, 'mlb_outcome_encoder.joblib')
joblib.dump(mlb_effect_subcat, 'mlb_effect_subcat_encoder.joblib')
joblib.dump(mlb_effect, 'mlb_effect_encoder.joblib')

