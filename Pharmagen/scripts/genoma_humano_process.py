# Script en Python para el procesamiento de genomas de VIH
# Autor: Astordna
# Fecha: 2024-06-15
# Descripción: Este script procesa los datos genómicos de VIH para análisis posteriores.
#              Almacena el resultado del procesamiento en un archivo de salida. El script
#              aisla las mutaciones relevantes y sus implicaciones farmacológicas en formato CSV.

import os
import pandas as pd
from Bio import SeqIO
import json
import Pharmagen

# Importa el diccionario de mutaciones

# Datos que generar. Seleccion de genomas y mutaciones

def procesar_genomas(input_genoma, MAIN_DIR):
    
    # Nombre del archivo que contiene el genoma del VIH secuenciado
    print("Asegúrese de que el archivo FASTA del genoma del VIH esté en la carpeta de trabajo. (Ruta actual: {})".format(os.getcwd()))
    filename = input("Introduce el nombre del archivo FASTA del genoma del VIH: ")
    filepath = MAIN_DIR + '/' + filename
    
    print("Procesando el archivo: {}".format(filename))
    
    num_sequences = sum(1 for _ in SeqIO.parse(filepath, "fasta"))
    
    # Si hay más de una secuencia, crear un archivo temporal por cada secuencia
    # Crear un directorio para las secuencias individuales en la carpeta cache
    
    if num_sequences > 1:
        print("Número de secuencias en el archivo: {}".format(num_sequences))
        print("Se creará un archivo temporal por cada secuencia en el archivo FASTA...")
        os.mkdir("{}/cache/{}}_sequences".format(MAIN_DIR, filename)) # Crear un directorio para las secuencias individuales en la carpeta cache 
        
    for seq in SeqIO.parse(filepath, "fasta"):
        SeqIO.write(seq, "{}/cache/{}sequence_{}.fasta", "fasta"\
            .format(MAIN_DIR, filename, seq.id))
    
    # Busqueda de coincidencias con las mutaciones del diccionario
    #for no se cuanto
        