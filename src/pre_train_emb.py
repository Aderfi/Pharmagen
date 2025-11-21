import pandas as pd

df = pd.read_csv('train_data/relationships_associated.tsv', sep='\t')

entity_columns = ['Entity1_name', 'Entity2_name']

final_vocab = set()

for col in entity_columns:
        if col in df.columns:
            # Obtén todos los valores únicos de la columna (ignorando NaNs)
            unique_entities = df[col].dropna().unique()
            
            # 4. Añade todas esas entidades únicas al set global
            # .update() añade todos los items de una lista al set,
            
            final_vocab.update(unique_entities)
        else:
            print(f"Advertencia: La columna '{col}' no se encontró. Saltando...")
    
print(f"Número total de entidades únicas encontradas: {len(final_vocab)}")
