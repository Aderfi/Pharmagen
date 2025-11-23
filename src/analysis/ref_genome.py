# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import os
import gzip
import shutil
import logging
import subprocess
import requests
from pathlib import Path
from datetime import datetime
from email.utils import parsedate_to_datetime
from tqdm.auto import tqdm

# Configuración del Proyecto
from src.cfg.config import REF_GENOME_FASTA, REF_GENOME_DIR

logger = logging.getLogger(__name__)

# URL oficial de Ensembl (GRCh38 Primary Assembly - sin haplotipos alternativos)
ENSEMBL_URL = "https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

class ReferenceGenomeManager:
    """
    Descarga, actualización e indexado del genoma de referencia. --> GChr38
    Asegura que existan los archivos .fa, .fai y los índices de BWA.
    """

    def __init__(self):
        self.target_fasta = REF_GENOME_FASTA
        self.download_dir = REF_GENOME_DIR
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        # Nombre del archivo gz temporal
        self.local_gz = self.download_dir / "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz"

    def _needs_download(self) -> bool:
        """Comprueba si el archivo remoto es más nuevo que el local."""
        if not self.local_gz.exists():
            return True

        try:
            response = requests.head(ENSEMBL_URL)
            if 'Last-Modified' in response.headers:
                remote_time = parsedate_to_datetime(response.headers['Last-Modified'])
                local_time = datetime.fromtimestamp(self.local_gz.stat().st_mtime).astimezone()
                
                if remote_time > local_time:
                    logger.info(f"Nueva versión detectada en Ensembl ({remote_time}).")
                    return True
            return False
        except Exception as e:
            logger.warning(f"No se pudo comprobar fecha remota: {e}. Omitiendo descarga.")
            return False

    def download_genome(self):
        """Descarga el genoma con barra de progreso."""
        if not self._needs_download() and self.target_fasta.exists():
            logger.info("El genoma local está actualizado.")
            return

        logger.info(f"Descargando genoma desde {ENSEMBL_URL}...")
        
        response = requests.get(ENSEMBL_URL, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(self.local_gz, 'wb') as f, tqdm(
            desc="Descargando GRCh38",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        self._decompress_genome()

    def _decompress_genome(self):
        """Descomprime el .gz directamente al nombre final HSapiens_GChr38.fa"""
        logger.info("Descomprimiendo genoma...")
        
        with gzip.open(self.local_gz, 'rb') as f_in:
            with open(self.target_fasta, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        logger.info(f"Genoma descomprimido en: {self.target_fasta}")

    def index_samtools(self):
        """Genera el índice .fai para acceso aleatorio."""
        fai_path = Path(str(self.target_fasta) + ".fai")
        if not fai_path.exists() or fai_path.stat().st_mtime < self.target_fasta.stat().st_mtime:
            logger.info("Generando índice Samtools (.fai)...")
            try:
                subprocess.run(f"samtools faidx {self.target_fasta}", shell=True, check=True)
                logger.info("\t- Índice .fai creado.")
            except subprocess.CalledProcessError:
                logger.error("XXXX -- Falló 'samtools faidx'. Asegúrate de tener samtools instalado.")
        else:
            logger.info("Índice .fai ya existe y está actualizado.")

    def index_bwa(self):
        """Genera los índices de BWA (.bwt, .pac, etc.) necesarios para el alineamiento."""
        # BWA genera varios archivos, verificamos si existe el principal (.bwt)
        bwt_path = Path(str(self.target_fasta) + ".bwt")
        
        if not bwt_path.exists():
            logger.info("Generando índice BWA (Esto puede tardar ~1 hora)...")
            try:
                # BWA index puede consumir mucha RAM, cuidado en máquinas pequeñas
                subprocess.run(f"bwa index {self.target_fasta}", shell=True, check=True)
                logger.info("✅ Índice BWA creado.")
            except subprocess.CalledProcessError:
                logger.error("❌ Falló 'bwa index'. Asegúrate de tener bwa instalado.")
        else:
            logger.info("Índice BWA ya existe.")

    def run(self):
        """Pipeline completo de descarga e indexado."""
        self.download_genome()
        
        # Verificamos que el fasta exista antes de indexar
        if self.target_fasta.exists():
            self.index_samtools()
            self.index_bwa()
        else:
            logger.error("No se encuentra el archivo FASTA para indexar.")

if __name__ == "__main__":
    # Configuración básica de log si se ejecuta solo
    logging.basicConfig(level=logging.INFO)
    manager = ReferenceGenomeManager()
    manager.run()