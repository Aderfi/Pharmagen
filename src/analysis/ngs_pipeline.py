# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import gzip
import requests

from src.cfg.manager import PROJECT_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
OS_NAME = platform.system().lower()

"""Funciones Auxiliares para el Pipeline NGS Farmacogenético."""

def set_ref_genome():
    """Descarga y prepara el genoma de referencia. (GChr38)"""
    # La ubicación del genoma se asume en data/ref_genome/
    logger.info("🔍 Verificando genoma de referencia...")

    if OS_NAME == "windows":
        logger.warning("[WARNING] Ejecutando en Windows nativo. Se recomienda usar WSL2 para compatibilidad total. Y mayor velocidad.")

        manager = GenomeManager(DATA_DIR / "ref_genome", GenomeManager.DEFAULT_CONFIG)
        manager.download_if_updated()
        manager.decompress_if_needed()
        manager.index_genome()


    elif OS_NAME in ["linux", "darwin"]:  # Linux o macOS
        script_sh = """
#!/bin/bash
set -e
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
"""
        subprocess.run(
            "sh", shell=True, input=script_sh.replace("$OUTPUT_DIR", str(DATA_DIR / "ref_genome")).encode(), check=True
        )
    else:
        logger.error(f"Sistema operativo no soportado: {OS_NAME}")
        sys.exit(1)

# ==============================================================================
# GENOMA DE REFERENCIA (Solo si Windows)
# ==============================================================================

class GenomeManager:
    """
    Controlador para la gestión del ciclo de vida de archivos genómicos:
    Descarga (con timestamp), Descompresión e Indexación.
    """
    DEFAULT_CONFIG = {
    "url": "https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "filename_gz": "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "filename_fa": "HSapiens_GChr38.fa"
    }

    def __init__(self, output_dir: str | Path, config: dict):
        self.output_dir = Path(output_dir)
        self.url = config["url"]
        self.local_gz = self.output_dir / config["filename_gz"]
        self.local_fa = self.output_dir / config["filename_fa"]
        self.index_fai = self.output_dir / (config["filename_fa"] + ".fai")

        # Asegurar que el directorio existe
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_remote_timestamp(self) -> float:
        """Obtiene el timestamp (Last-Modified) del servidor remoto."""
        try:
            response = requests.head(self.url, allow_redirects=True)
            if 'Last-Modified' in response.headers:
                dt = parsedate_to_datetime(response.headers['Last-Modified'])
                return dt.timestamp()
            return time.time() 
        except requests.RequestException as e:
            logger.info(f"Error conectando al servidor: {e}")
            sys.exit(1)

    def download_if_updated(self):
        """
        Replica el comportamiento de 'wget --timestamping'.
        Solo descarga si el archivo local no existe o el remoto es más nuevo.
        """
        logger.info(f"Comprobando actualizaciones para: {self.local_gz.name}...")

        remote_mtime = self._get_remote_timestamp()
        should_download = False

        if not self.local_gz.exists():
            should_download = True
            logger.info("El archivo local no existe. Iniciando descarga...")
        else:
            local_mtime = self.local_gz.stat().st_mtime
            # Comparamos timestamps
            if remote_mtime > local_mtime:
                should_download = True
                logger.info("Se ha detectado una nueva versión en el servidor.")
            else:
                logger.info("El archivo comprimido local ya está actualizado.")

        if should_download:
            try:
                with requests.get(self.url, stream=True) as r:
                    r.raise_for_status()
                    with open(self.local_gz, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)

                os.utime(self.local_gz, (time.time(), remote_mtime))
                logger.info("Descarga completada.")
            except Exception as e:
                logger.info(f"Error durante la descarga: {e}")
                sys.exit(1)

    def decompress_if_needed(self):
        """
        Si el .fa no existe O el .gz es más nuevo que el .fa, descomprimir.
        """
        should_decompress = False

        if not self.local_gz.exists():
            logger.info("Error: No se encuentra el archivo .gz para descomprimir.")
            return

        if not self.local_fa.exists():
            should_decompress = True
            logger.info("Falta el archivo descomprimido (.fa).")
        elif self.local_gz.stat().st_mtime > self.local_fa.stat().st_mtime:
            should_decompress = True
            logger.info("El archivo comprimido es más reciente que el descomprimido.")

        if should_decompress:
            logger.info("Descomprimiendo (esto puede tardar unos minutos)...")
            try:
                # Lectura por bloques para no saturar la RAM con genomas grandes
                with gzip.open(self.local_gz, 'rb') as f_in:
                    with open(self.local_fa, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                logger.info("Descompresión finalizada.")
            except Exception as e:
                logger.info(f"Error al descomprimir: {e}")
                sys.exit(1)

    def index_genome(self):
        """
        Verifica si el índice existe. Si no, o si se acaba de actualizar el genoma, indexa.
        Utiliza subprocess para llamar a samtools.
        """
        # Si acabamos de descomprimir, el .fa será más nuevo que el .fai, 
        # o si el .fai no existe.
        should_index = False

        if not self.local_fa.exists():
            logger.info("No se puede indexar: falta el archivo FASTA.")
            return

        if not self.index_fai.exists():
            should_index = True
            logger.info("El archivo de índice no existe.")
        elif self.local_fa.stat().st_mtime > self.index_fai.stat().st_mtime:
            should_index = True
            logger.info("El genoma es más reciente que su índice.")

        if should_index:
            logger.info("Indexando con samtools...")

            # Verificación de dependencia
            if not shutil.which("samtools"):
                logger.info("ERROR CRÍTICO: 'samtools' no está instalado o no está en el PATH.")
                logger.info("En Debian 13: sudo apt install samtools")
                sys.exit(1)

            try:
                # check=True lanza una excepción si el comando falla
                subprocess.run(["samtools", "faidx", str(self.local_fa)], check=True)
                logger.info("\n Índice creado/actualizado correctamente. \n")
            except subprocess.CalledProcessError as e:
                logger.info(f"Error al ejecutar samtools: {e}")
                sys.exit(1)
        else:
             logger.info("El índice ya está actualizado.")

# =============================================================================
# CLASE BASE PARA HERRAMIENTAS BIOINFORMÁTICAS EXTERNAS
# =============================================================================

class BioToolExecutor:
    """
    Clase base herramientas bioinformáticas externas (CLI wrappers).
    Maneja subprocess, logging y captura de errores.
    """
    def __init__(self, threads: int = 4):
        self.threads = str(threads)

    def _run_cmd(self, command: str, description: str):
        logger.info(f"🚀 Iniciando: {description}")
        logger.debug(f"CMD: {command}")

        # Detección de sistema operativo para advertencias
        if sys.platform == "win32":
            logger.warning("[WARNING] Ejecutando pipeline bioinformático en Windows nativo.")
            logger.warning("Si fallan los pipes (|) o no encuentra herramientas, usa WSL2. !!!!")

        try:
            process = subprocess.run(
                command,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT)
            )

            logger.info(f"✅ Finalizado: {description}")
            return process

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error crítico en {description}")
            logger.error(f"Código de salida: {e.returncode}")
            # Capturamos tanto stdout como stderr
            if e.stdout:
                logger.error(f"Salida estándar (últimas líneas):\n{e.stdout[-500:]}")
            if e.stderr:
                logger.error(f"Salida de error:\n{e.stderr}")

            raise RuntimeError(f"Fallo en el pipeline bioinformático ({description}). Ver logs para detalles.")

# ==============================================================================
# FASE 1: PROCESAMIENTO DE LECTURAS CRUDAS
# ==============================================================================

class ProcessRawGenome(BioToolExecutor):
    """
    Fase 1: Quality Control & Trimming.
    Herramientas: FastQC, FastP.
    """
    def __init__(self, output_dir: Path, threads: int = 4):
        super().__init__(threads)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_fastqc(self, fastq_files: list[Path], step_name: str = "pre_qc"):
        """Ejecuta FastQC para análisis de calidad."""
        out_dir = self.output_dir / step_name
        out_dir.mkdir(exist_ok=True)

        files_str = " ".join([str(f) for f in fastq_files])
        cmd = f"fastqc -t {self.threads} -o {out_dir} {files_str}"

        self._run_cmd(cmd, f"FastQC ({step_name})")
        return out_dir

    def run_fastp(self, r1: Path, r2: Path, sample_name: str) -> dict[str, Path]:
        """Ejecuta FastP para limpieza de adaptadores y calidad."""
        clean_dir = self.output_dir / "clean_reads"
        clean_dir.mkdir(exist_ok=True)

        out_r1 = clean_dir / f"{sample_name}_R1_clean.fastq.gz"
        out_r2 = clean_dir / f"{sample_name}_R2_clean.fastq.gz"
        report_html = clean_dir / f"{sample_name}_fastp.html"
        report_json = clean_dir / f"{sample_name}_fastp.json"

        cmd = (
            f"fastp -i {r1} -I {r2} -o {out_r1} -O {out_r2} "
            f"--detect_adapter_for_pe -w {self.threads} "
            f"-h {report_html} -j {report_json}"
        )

        self._run_cmd(cmd, f"FastP Cleaning ({sample_name})")
        return {"r1": out_r1, "r2": out_r2}

# ==============================================================================
# FASE 2: MAPEO Y ALINEAMIENTO
# ==============================================================================

class MappingAlignmentAnalysis(BioToolExecutor):
    """
    Fase 2: Alineamiento a Referencia.
    Herramientas: BWA, Samtools, Picard, Qualimap.
    """
    def __init__(self, output_dir: Path, ref_genome: Path = REF_GENOME_FASTA, threads: int = 8):
        super().__init__(threads)
        self.output_dir = Path(output_dir)
        self.ref_genome = ref_genome
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._check_bwa_index()

    def _check_bwa_index(self):
        if not Path(str(self.ref_genome) + ".bwt").exists():
            logger.warning("⚠️ Índice BWA no encontrado. Creando (esto puede tardar)...")
            self._run_cmd(f"bwa index {self.ref_genome}", "Indexado BWA")

    def map_reads(self, r1: Path, r2: Path, sample_name: str) -> Path:
        """Mapea con BWA-MEM y ordena con Samtools."""
        bam_dir = self.output_dir / "bams"
        bam_dir.mkdir(exist_ok=True)
        raw_bam = bam_dir / f"{sample_name}_sorted.bam"

        # Read Group es obligatorio para herramientas downstream
        rg_tag = f"@RG\\tID:{sample_name}\\tSM:{sample_name}\\tPL:ILLUMINA"

        # Pipe optimization: BWA -> Samtools Sort
        cmd = (
            f"bwa mem -t {self.threads} -R \"{rg_tag}\" {self.ref_genome} {r1} {r2} | "
            f"samtools sort -@ {self.threads} -o {raw_bam} -"
        )

        self._run_cmd(cmd, f"BWA Alignment ({sample_name})")
        self._run_cmd(f"samtools index {raw_bam}", "Indexado BAM")
        return raw_bam

    def preprocess_identify_duplicates(self, input_bam: Path, sample_name: str) -> Path:
        """Identifica duplicados de PCR con Picard."""
        dedup_bam = self.output_dir / "bams" / f"{sample_name}_dedup.bam"
        metrics = self.output_dir / "bams" / f"{sample_name}_dedup_metrics.txt"

        cmd = (
            f"picard MarkDuplicates I={input_bam} O={dedup_bam} M={metrics} "
            "REMOVE_DUPLICATES=false VALIDATION_STRINGENCY=LENIENT"
        )

        self._run_cmd(cmd, "Picard MarkDuplicates")
        self._run_cmd(f"samtools index {dedup_bam}", "Indexado Dedup BAM")
        return dedup_bam

    def quality_analysis(self, bam_file: Path):
        """Analiza la calidad del BAM final con Qualimap."""
        qm_dir = self.output_dir / "qualimap_report"
        # Usamos try/except porque Qualimap a veces falla en entornos sin X11 (headless)
        try:
            self._run_cmd(
                f"qualimap bamqc -bam {bam_file} -outdir {qm_dir} --java-mem-size=4G", 
                "Qualimap BamQC"
            )
        except RuntimeError:
            logger.warning("Qualimap falló (posible error de GUI). Continuando pipeline.")

# ==============================================================================
# FASE 3: IDENTIFICACIÓN Y ANÁLISIS DE VARIANTES
# ==============================================================================

class VariantIdentificationAnalysis(BioToolExecutor):
    """
    Fase 3: Variant Calling.
    Herramientas: Freebayes, VCFtools.
    """
    def __init__(self, output_dir: Path, ref_genome: Path = REF_GENOME_FASTA):
        super().__init__(threads=1) # Freebayes no escala bien por threads
        self.output_dir = Path(output_dir)
        self.ref_genome = ref_genome
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def identify_variants(self, bam_file: Path, sample_name: str) -> Path:
        """Llama variantes con Freebayes."""
        vcf_raw = self.output_dir / f"{sample_name}_raw.vcf"
        cmd = f"freebayes -f {self.ref_genome} {bam_file} > {vcf_raw}"
        self._run_cmd(cmd, "Freebayes Variant Calling")
        return vcf_raw

    def filter_variants(self, input_vcf: Path, sample_name: str) -> Path:
        """Filtra variantes de baja calidad."""
        vcf_filtered = self.output_dir / f"{sample_name}_filtered.vcf"

        # Filtros estándar clínicos: Calidad > 20, Profundidad > 10
        cmd = (
            f"vcftools --vcf {input_vcf} --minQ 20 --minDP 10 "
            f"--recode --recode-INFO-all --out {self.output_dir / sample_name}_temp"
        )

        self._run_cmd(cmd, "VCFtools Filtering")

        # Renombrar salida de vcftools (.recode.vcf)
        temp_out = self.output_dir / f"{sample_name}_temp.recode.vcf"
        if temp_out.exists():
            shutil.move(str(temp_out), str(vcf_filtered))

        return vcf_filtered

# ==============================================================================
# FASE 4: ANOTACIÓN (VEP)
# ==============================================================================

class VariantAnnotator(BioToolExecutor):
    """
    Fase 4: Anotación funcional.
    Herramienta: VEP (Variant Effect Predictor).
    Requisito: VEP instalado y caché configurada en ~/.vep
    """
    def __init__(self, output_dir: Path, threads: int = 4, assembly: str = "GRCh38"):
        super().__init__(threads)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assembly = assembly

    def annotate_variants(self, input_vcf: Path, sample_name: str) -> Path:
        """
        Ejecuta VEP para anotar el VCF.
        Genera un VCF anotado y un reporte HTML.
        """
        annotated_vcf = self.output_dir / f"{sample_name}_annotated.vcf"
        stats_file = self.output_dir / f"{sample_name}_vep_summary.html"

        # Comando VEP estándar para farmacogenética
        # --cache: usa caché local (rápido)
        # --offline: no conecta a internet (privacidad)
        # --pick: elige la transcripción canónica más severa por gen
        cmd = (
            f"vep -i {input_vcf} -o {annotated_vcf} "
            f"--assembly {self.assembly} "
            f"--cache --offline --force_overwrite "
            f"--vcf --stats_file {stats_file} "
            f"--pick --fork {self.threads}"
        )

        self._run_cmd(cmd, "VEP Annotation")
        return annotated_vcf

# ==============================================================================
# ORQUESTADOR PRINCIPAL
# ==============================================================================

def run_full_ngs_pipeline(r1: Path, r2: Path, sample_name: str):
    """
    Ejecuta todas las fases del pipeline secuencialmente.
    """
    # Directorio base para este paciente en processed/
    base_results = Path(DATA_DIR) / "processed" / sample_name

    logger.info(f"\n🧬 Iniciando Pipeline Farmacogenético para: {sample_name}")
    logger.info("="*70)

    try:
        # 1. Process Raw Genome
        step1 = ProcessRawGenome(base_results / "01_qc")
        step1.run_fastqc([r1, r2], "raw_fastqc")
        clean_files = step1.run_fastp(r1, r2, sample_name)
        step1.run_fastqc([clean_files["r1"], clean_files["r2"]], "clean_fastqc")

        # 2. Mapping & Alignment
        step2 = MappingAlignmentAnalysis(base_results / "02_alignment")
        raw_bam = step2.map_reads(clean_files["r1"], clean_files["r2"], sample_name)
        final_bam = step2.preprocess_identify_duplicates(raw_bam, sample_name)
        step2.quality_analysis(final_bam)

        # 3. Variant Identification
        step3 = VariantIdentificationAnalysis(base_results / "03_variants")
        raw_vcf = step3.identify_variants(final_bam, sample_name)
        filtered_vcf = step3.filter_variants(raw_vcf, sample_name)

        # 4. Annotation
        step4 = VariantAnnotator(base_results / "04_annotation")
        final_vcf = step4.annotate_variants(filtered_vcf, sample_name)

        logger.info(f"\n✅ Pipeline completado exitosamente.")
        logger.info(f"📂 VCF Anotado Final: {final_vcf}")
        logger.info(f"📊 Reporte VEP: {base_results / '04_annotation' / f'{sample_name}_vep_summary.html'}")

    except Exception as e:
        logger.critical(f"El pipeline falló: {e}")
        logger.info(f"\n❌ Error crítico en el pipeline: {e}")

def main():
    subprocess.run(["", ""], check=True)
    ...

if __name__ == "__main__":
    main()