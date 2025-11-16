#!/bin/bash

# Genome Download Script. Checks version and downloads if updated
# Usage: GDown.sh <genome_name> <version> <output_directory>

URL_passembly="https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
Local_Genome="data/Ref_Genome/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
LOCAL_GZ="$OUTPUT_DIR/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"
LOCAL_FA="$OUTPUT_DIR/HSapiens_GChr38.fa"
INDEX_FAI="$OUTPUT_DIR/HSapiens_GChr38.fa.fai"


echo "Comprobando actualizaciones del genoma de referencia..."

wget --timestamping --directory-prefix="$OUTPUT_DIR" "$URL_passembly"

if [ -f "$LOCAL_GZ" ]; then
    # Si no existe el .fa O el .gz es más nuevo que el .fa
    if [ ! -f "$LOCAL_FA" ] || [ "$LOCAL_GZ" -nt "$LOCAL_FA" ]; then
        echo "Se ha detectado una nueva versión o falta el archivo descomprimido."
        echo "Descomprimiendo..."
        # -k mantiene el archivo .gz original para futuras comparaciones de timestamp
        gunzip -k -f "$LOCAL_GZ" 
        
        echo "Indexando..."

        samtools faidx "$LOCAL_FA"

        echo -e "\n Genoma actualizado e indexado correctamente. \n"
        echo -e "\n"
    else
        echo "El genoma local ya está actualizado."
    fi
fi

if [ ! -f "$INDEX_FAI" ]; then
    echo "El archivo de índice no existe. Indexando..."
    samtools faidx "$LOCAL_FA"
    echo -e "\n Índice creado correctamente. \n"
fi





    