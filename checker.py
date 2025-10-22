import pandas as pd
import re

df = pd.read_csv("train_therapeutic_outcome.csv", sep=';')

for col in df.columns:
    field_have_multi_values = False
    for idx, value in df[col].items():
        if isinstance(value, str) and ',' in value:
            field_have_multi_values = True
            print(f"Columna: {col} Tiene múltiples valores")
            break

print(df['Effect'].unique())