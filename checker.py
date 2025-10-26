import pandas as pd
import re
import json


df = pd.read_csv("train_therapeutic_outcome_cleaned.csv", sep=';', index_col=False)

df['Effect'] = df['Effect_direction'] + "_" + df['Effect']
df.pop('Effect_direction')

df.to_csv("Reorden_train_therapeutic_outcome_cleaned_effect_direction.csv", sep=';', index=False)