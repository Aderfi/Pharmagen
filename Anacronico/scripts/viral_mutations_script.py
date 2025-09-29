# Este scipt tiene dos objetivos principales:
# 1. Definir y almacenar un diccionario de mutaciones virales relevantes para el análisis de genomas de VIH.
#       El almacenamiento se realiza mediante la importación de bases de datos oficiales
# 2. Guardar este diccionario en un archivo CSV para su uso posterior en el análisis de genomas.
# Autor: Astordna
# Fecha: 2024-06-15

import os
import pandas as pd
from Bio import SeqIO  # Importa el directorio principal
import Anacronico 

# Ruta al archivo CSV donde se almacenarán las mutaciones
output_path = os.path.join(os.path.dirname(__file__), 'viral_mutations.csv')

# Cada mutación será un diccionario con claves: 'Gen', 'Cadena de nucleótidos', 'Implicacion Farmacologica'
datos_mutaciones = [12*[None] for _ in range(0)]  # Lista para almacenar los datos de las mutaciones


mutations_df = pd.DataFrame(datos_mutaciones)

# Crear el archivo CSV si no existe
if not os.path.exists(output_path):
    mutations_df.to_csv(output_path, index=False)


# Guardar el DataFrame en el archivo CSV en modo append
mutations_df.to_csv(output_path, mode='a', index=False)
    