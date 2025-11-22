# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import Bio
from pathlib import Path
from Bio import SeqIO
import pandas as pd
import csv
import sys
import os
import glob
import namex


### Importación de configuraciones y rutas desde el archivo config.py ###
from src.cfg.config import *


def increase_csv_field_limit():
    """Aumenta el límite de tamaño de campo para el lector CSV de Python."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 2)


# Definicion del proceso principal para el analisis de una genoteca
"""
--------------------------------------------------------------
Fase 1: Procesamiento de Lecturas Crudas (Color Púrpura)
--------------------------------------------------------------
    ### Secuenciador
            input: genoteca
            output: archivos FASTQ-raw

    ### Análisis de calidad
            input: archivos FASTQ-raw
            output: archivos HTML
            herramienta: FastQC

    ### Limpieza de adaptadores y regiones de baja calidad
            input: archivos FASTQ-raw
            output: archivos FASTQ-clean
            herramienta: FastP

    ### Revisión de calidad posprocesado
            input: archivos FASTQ-clean
            output: archivos HTML
            herramienta: FastQC
            
--------------------------------------------------------------
Fase 2: Mapeo y Procesamiento de Alineamientos (Color Gris)
--------------------------------------------------------------
    Mapeo a genoma de referencia
        input: archivos FASTQ-clean + genoma referencia FASTA
        output: archivo SAM
        herramienta: BWA

    Conversión de archivos y procesamiento
        input: SAM
        output: BAM/BAI
        herramienta: SAMTOOLS

    Análisis de la calidad del mapeo
        input: BAM
        output: archivo HTML/PDF
        herramienta: Qualimap

    Preprocesado: identificación de duplicados
        input: BAM
        output: BAM
        herramienta: PicardTools
    
--------------------------------------------------------------
Fase 3: Identificación y Análisis de Variantes (Color Naranja)
--------------------------------------------------------------
    Identificación de variantes

        input: BAM

        output: VCF

        herramienta: Freebayes

    Filtrado de SNP/SNV y de indels

        input: VCF

        output: VCF

        herramienta: VCFtools

    Visualización de variantes

        input: VCF

        output: visual

        herramienta: IGV

    Anotación de variantes

        input: VCF

        output: informe escrito

        herramienta: VEP/Annovar
--------------------------------------------------------------
Fase 4: Conclusión (Color Rojo)
--------------------------------------------------------------
    Interpretación de los resultados
"""


class ProcessRawGenome:
    def __init__(self, raw_fastq_dir, processed_fastq_dir, cache_dir):
        self.raw_fastq_dir = raw_fastq_dir
        self.processed_fastq_dir = processed_fastq_dir
        self.cache_dir = cache_dir

    def run_fastqc(self):
        # Implementar análisis de calidad usando FastQC
        pass

    def run_fastp(self):
        # Implementar limpieza de adaptadores y regiones de baja calidad usando FastP
        pass

    def run_postprocess_fastqc(self):
        # Implementar revisión de calidad posprocesado usando FastQC
        pass


class MappingAlignmentAnalysis:
    def __init__(self, raw_fastq_dir, reference_genome, processed_bam_dir, cache_dir):
        self.raw_fastq_dir = raw_fastq_dir
        self.reference_genome = reference_genome
        self.processed_bam_dir = processed_bam_dir
        self.cache_dir = cache_dir

    def map_reads(self):
        # Implementar mapeo usando BWA
        pass

    def convert_and_process_sam(self):
        # Implementar conversión y procesamiento usando SAMTOOLS
        pass

    def quality_analysis(self):
        # Implementar análisis de calidad usando Qualimap
        pass

    def preprocess_identify_duplicates(self):
        # Implementar identificación de duplicados usando PicardTools
        pass


class VariantIdentificationAnalysis:
    def __init__(self, processed_bam_dir, processed_vcf_dir, cache_dir):
        self.processed_bam_dir = processed_bam_dir
        self.processed_vcf_dir = processed_vcf_dir
        self.cache_dir = cache_dir

    def identify_variants(self):
        # Implementar identificación de variantes usando Freebayes
        pass

    def filter_variants(self):
        # Implementar filtrado de variantes usando VCFtools
        pass

    def visualize_variants(self):
        # Implementar visualización de variantes usando IGV
        pass

    def annotate_variants(self):
        # Implementar anotación de variantes usando VEP/Annovar
        pass


"""
ESTABLECER FLUJO DE TRABAJO PRINCIPAL Y GUARDAR EN DATA/PROCESSED
"""
