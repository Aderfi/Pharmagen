# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

import argparse
import gzip
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

try:
    from src.cfg.manager import PROJECT_ROOT
except ImportError:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from src.interface.ui import ConsoleIO, ProgressBar

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Pharmagen-NGS")
cli = ConsoleIO()

# ==============================================================================
# CONSTANTES Y CONFIGURACIÓN GLOBAL
# ==============================================================================

DATA_DIR = PROJECT_ROOT / "data"
REF_GENOME_DIR = DATA_DIR / "ref_genome"
REF_GENOME_FASTA = REF_GENOME_DIR / "HSapiens_GChr38.fa"

GENOME_CONFIG = {
    "url": "https://ftp.ensembl.org/pub/release-114/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "filename_gz": "Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz",
    "filename_fa": "HSapiens_GChr38.fa"
}

# ==============================================================================
# GESTOR DE GENOMA DE REFERENCIA (Cross-Platform)
# ==============================================================================

class GenomeManager:
    """
    Controlador para la gestión del ciclo de vida de archivos genómicos:
    Descarga, Descompresión e Indexación. Funciona en Linux, Mac y Windows.
    """
    def __init__(self, output_dir: Path, config: dict):
        self.output_dir = output_dir
        self.url = config["url"]
        self.local_gz = self.output_dir / config["filename_gz"]
        self.local_fa = self.output_dir / config["filename_fa"]
        self.index_fai = self.output_dir / (config["filename_fa"] + ".fai")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_genome(self):
        """Verificación y preparación del genoma de referencia."""
        logger.info("🔍 Verificando estado del genoma de referencia...")
        self._download_if_needed()
        self._decompress_if_needed()
        self._index_genome()
        return self.local_fa

    def _get_remote_timestamp(self) -> float:
        try:
            response = requests.head(self.url, allow_redirects=True)
            if 'Last-Modified' in response.headers:
                dt = parsedate_to_datetime(response.headers['Last-Modified'])
                return dt.timestamp()
            return time.time()
        except requests.RequestException as e:
            logger.warning(f"No se pudo verificar fecha remota: {e}")
            return time.time()

    def _download_if_needed(self):
        remote_mtime = self._get_remote_timestamp()
        should_download = False

        if not self.local_gz.exists():
            should_download = True
            ConsoleIO.print_warning("📥 El genoma no existe localmente. Iniciando descarga...")
        elif remote_mtime > self.local_gz.stat().st_mtime:
            should_download = True
            cli.print_info("Nueva versión detectada en servidor.")

        if should_download:
            try:
                with requests.get(self.url, stream=True) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))
                    downloaded = 0

                    cli.print_info(f"Descargando {self.url}...")
                    with open(self.local_gz, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Feedback simple de progreso
                            if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                                print(f"Descargado: {downloaded / (1024*1024):.1f} MB", end='\r')
                    print() # Salto de línea al terminar

                # Actualizar timestamp local para coincidir con remoto
                os.utime(self.local_gz, (time.time(), remote_mtime))
                logger.info("✅ Descarga completada.")
            except Exception as e:
                logger.error(f"❌ Error crítico descargando genoma: {e}")
                sys.exit(1)

    def _decompress_if_needed(self):
        # Descomprimir si falta el .fa o si el .gz es más nuevo
        if not self.local_fa.exists() or (self.local_gz.exists() and self.local_gz.stat().st_mtime > self.local_fa.stat().st_mtime):
            cli.print_info("Descomprimiendo genoma (esto puede tardar)...")
            try:
                with gzip.open(self.local_gz, 'rb') as f_in:
                    with open(self.local_fa, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                cli.print_success("Descompresión finalizada.")
            except Exception as e:
                cli.print_error(f"Error descomprimiendo: {e}")
                sys.exit(1)

    def _index_genome(self):
        # Indexar si falta el .fai o si el .fa es más nuevo
        if not self.index_fai.exists() or self.local_fa.stat().st_mtime > self.index_fai.stat().st_mtime:
            cli.print_info("Generando índice FAI con samtools...")
            if not shutil.which("samtools"):
                cli.print_error("'samtools' no encontrado. Instálalo: sudo apt install samtools")
                sys.exit(1)

            try:
                subprocess.run(["samtools", "faidx", str(self.local_fa)], check=True)
                cli.print_success("Índice creado.")
            except subprocess.CalledProcessError as e:
                cli.print_error(f"Fallo al indexar: {e}")
                sys.exit(1)

# ==============================================================================
# CLASE BASE (WRAPPER DE HERRAMIENTAS)
# ==============================================================================

class BioToolExecutor:
    """Clase base para ejecutar comandos de shell de forma segura."""
    def __init__(self, threads: int = 4):
        self.threads = str(threads)

    def _run_cmd(self, command: str, description: str):
        cli.print_info(f"Ejecutando: {description}", "🔥")
        logger.debug(f"CMD: {command}")

        try:
            # Usamos executable='/bin/bash' en Linux para soportar pipes (|) correctamente
            shell_exec = '/bin/bash' if platform.system() != 'Windows' else None

            process = subprocess.run(
                command,
                shell=True,
                check=True,
                executable=shell_exec,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return process

        except subprocess.CalledProcessError as e:
            cli.print_error(f"Error en {description}")
            logger.error(f"Exit Code: {e.returncode}")
            if e.stderr:
                logger.error(f"STDERR: {e.stderr.strip()[-1000:]}") # Últimos 1000 caracteres
            raise RuntimeError(f"Fallo en paso bioinformático: {description}")

# ==============================================================================
# FASE 1: PROCESAMIENTO DE LECTURAS (QC & TRIMMING)
# ==============================================================================

class ProcessRawGenome(BioToolExecutor):
    def __init__(self, output_dir: Path, threads: int = 4):
        super().__init__(threads)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_fastqc(self, fastq_files: list[Path], step_name: str = "pre_qc"):
        out_dir = self.output_dir / step_name
        out_dir.mkdir(exist_ok=True)

        # Validar existencia de archivos
        files_str = " ".join([str(f) for f in fastq_files if f.exists()])
        if not files_str:
            logger.warning(f"No se encontraron archivos para FastQC ({step_name})")
            return

        cmd = f"fastqc -t {self.threads} -o {out_dir} {files_str}"
        self._run_cmd(cmd, f"FastQC ({step_name})")

    def run_fastp(self, r1: Path, r2: Path, sample_name: str) -> dict[str, Path]:
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
# FASE 2: ALINEAMIENTO (MAPPING)
# ==============================================================================

class MappingAlignmentAnalysis(BioToolExecutor):
    def __init__(self, output_dir: Path, ref_genome: Path, threads: int = 8):
        super().__init__(threads)
        self.output_dir = output_dir
        self.ref_genome = ref_genome
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._check_bwa_index()

    def _check_bwa_index(self):
        # BWA genera varios archivos (.bwt, .pac, etc). Comprobamos uno.
        if not Path(str(self.ref_genome) + ".bwt").exists():
            logger.info("⚠️ Índice BWA no encontrado. Generando (esto tarda)...")
            self._run_cmd(f"bwa index {self.ref_genome}", "Indexado BWA")

    def map_reads(self, r1: Path, r2: Path, sample_name: str) -> Path:
        bam_dir = self.output_dir / "bams"
        bam_dir.mkdir(exist_ok=True)
        sorted_bam = bam_dir / f"{sample_name}_sorted.bam"

        rg_tag = f"@RG\\tID:{sample_name}\\tSM:{sample_name}\\tPL:ILLUMINA"

        # Pipeline: BWA MEM -> Samtools Sort
        cmd = (
            f"bwa mem -t {self.threads} -R \"{rg_tag}\" {self.ref_genome} {r1} {r2} | "
            f"samtools sort -@ {self.threads} -o {sorted_bam} -"
        )

        self._run_cmd(cmd, f"Alineamiento BWA ({sample_name})")
        self._run_cmd(f"samtools index {sorted_bam}", "Indexado BAM")
        return sorted_bam

    def mark_duplicates(self, input_bam: Path, sample_name: str) -> Path:
        dedup_bam = self.output_dir / "bams" / f"{sample_name}_dedup.bam"
        metrics = self.output_dir / "bams" / f"{sample_name}_dedup_metrics.txt"

        cmd = (
            f"picard MarkDuplicates I={input_bam} O={dedup_bam} M={metrics} "
            "REMOVE_DUPLICATES=false VALIDATION_STRINGENCY=LENIENT"
        )
        self._run_cmd(cmd, "Picard MarkDuplicates")
        self._run_cmd(f"samtools index {dedup_bam}", "Indexado Dedup BAM")
        return dedup_bam

# ==============================================================================
# FASE 3: VARIANT CALLING
# ==============================================================================

class VariantIdentificationAnalysis(BioToolExecutor):
    def __init__(self, output_dir: Path, ref_genome: Path):
        # Freebayes no paraleliza bien nativamente, usamos 1 thread o parallel externo
        super().__init__(threads=1)
        self.output_dir = output_dir
        self.ref_genome = ref_genome
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def call_variants(self, bam_file: Path, sample_name: str) -> Path:
        vcf_raw = self.output_dir / f"{sample_name}_raw.vcf"
        # Freebayes standard call
        cmd = f"freebayes -f {self.ref_genome} {bam_file} > {vcf_raw}"
        self._run_cmd(cmd, "Freebayes Caller")
        return vcf_raw

    def filter_variants(self, input_vcf: Path, sample_name: str) -> Path:
        vcf_filtered = self.output_dir / f"{sample_name}_filtered.vcf"
        prefix = self.output_dir / f"{sample_name}_temp"

        # Filtros básicos: Calidad > 20, Profundidad > 10
        cmd = (
            f"vcftools --vcf {input_vcf} --minQ 20 --minDP 10 "
            f"--recode --recode-INFO-all --out {prefix}"
        )
        self._run_cmd(cmd, "VCFtools Filter")

        # vcftools añade sufijo .recode.vcf
        recode_out = Path(f"{prefix}.recode.vcf")
        if recode_out.exists():
            shutil.move(recode_out, vcf_filtered)
        else:
            raise FileNotFoundError(f"VCFtools no generó salida: {recode_out}")

        return vcf_filtered

# ==============================================================================
# FASE 4: ANOTACIÓN (VEP)
# ==============================================================================

class VariantAnnotator(BioToolExecutor):
    def __init__(self, output_dir: Path, threads: int = 4, assembly: str = "GRCh38"):
        super().__init__(threads)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.assembly = assembly

    def run_vep(self, input_vcf: Path, sample_name: str) -> Path:
        annotated_vcf = self.output_dir / f"{sample_name}_annotated.vcf"
        stats_file = self.output_dir / f"{sample_name}_vep.html"

        # VEP: Cache local, offline, coge la variante más severa
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
# ORQUESTADOR (PIPELINE RUNNER)
# ==============================================================================

def run_pipeline(r1: Path, r2: Path, sample_name: str, threads: int):
    # 0. Preparar Genoma
    genome_mgr = GenomeManager(REF_GENOME_DIR, GENOME_CONFIG)
    ref_genome_path = genome_mgr.prepare_genome()

    # Directorio de resultados
    results_dir = DATA_DIR / "processed" / sample_name
    logger.info(f"📁 Directorio de salida: {results_dir}")

    try:
        # 1. QC & Cleaning
        step1 = ProcessRawGenome(results_dir / "01_qc", threads)
        step1.run_fastqc([r1, r2], "raw")
        clean = step1.run_fastp(r1, r2, sample_name)
        step1.run_fastqc([clean["r1"], clean["r2"]], "clean")

        # 2. Alignment
        step2 = MappingAlignmentAnalysis(results_dir / "02_alignment", ref_genome_path, threads)
        raw_bam = step2.map_reads(clean["r1"], clean["r2"], sample_name)
        final_bam = step2.mark_duplicates(raw_bam, sample_name)

        # 3. Variant Calling
        step3 = VariantIdentificationAnalysis(results_dir / "03_variants", ref_genome_path)
        raw_vcf = step3.call_variants(final_bam, sample_name)
        filtered_vcf = step3.filter_variants(raw_vcf, sample_name)

        # 4. Annotation
        step4 = VariantAnnotator(results_dir / "04_annotation", threads)
        final_vcf = step4.run_vep(filtered_vcf, sample_name)

        logger.info("="*60)
        logger.info(f"✅ PIPELINE COMPLETADO EXITOSAMENTE PARA: {sample_name}")
        logger.info(f"📄 Resultado final: {final_vcf}")
        logger.info("="*60)

    except Exception as e:
        logger.critical(f"❌ El pipeline se detuvo por error: {e}")
        sys.exit(1)

# ==============================================================================
# MAIN (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pharmagen NGS Pipeline")
    parser.add_argument("-r1", "--read1", required=True, type=Path, help="Ruta al archivo FASTQ R1 (Forward)")
    parser.add_argument("-r2", "--read2", required=True, type=Path, help="Ruta al archivo FASTQ R2 (Reverse)")
    parser.add_argument("-n", "--name", required=True, type=str, help="ID de la muestra (ej. Patient_01)")
    parser.add_argument("-t", "--threads", type=int, default=4, help="Número de hilos CPU (default: 4)")

    args = parser.parse_args()

    if not args.read1.exists() or not args.read2.exists():
        logger.error("❌ No se encuentran los archivos de entrada.")
        sys.exit(1)

    run_pipeline(args.read1, args.read2, args.name, args.threads)

if __name__ == "__main__":
    main()
