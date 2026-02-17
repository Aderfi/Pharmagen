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

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pysam  # type: ignore / GNU/Linux

from src.analysis.ngs_pipeline import PROJECT_ROOT, REF_GENOME_FASTA
from src.interface.ui import ConsoleIO

# Rutas constantes
RUTA_FASTA = REF_GENOME_FASTA
RUTA_VCF_DIR = PROJECT_ROOT / "data"


def seleccionar_vcf(vcf_path: Path = RUTA_VCF_DIR) -> str | Path | None:
    """
    Lista y permite seleccionar un archivo VCF.gz del directorio.
    """
    vcf_files = list(vcf_path.glob("*.vcf.gz"))

    if not vcf_files:
        ConsoleIO.print_error(f"No se encontraron archivos .vcf.gz en {vcf_path}")
        return None

    ConsoleIO.print_info("\n Pacientes disponibles:")
    for i, vcf_file in enumerate(vcf_files):
        print(f" -- {i + 1}. {vcf_file.name.replace('.vcf.gz', '')}")

    while True:
        try:
            entrada = input("\nSelecciona el número del paciente (o 'q' para salir): ")
            if entrada.lower() == "q":
                return None

            seleccion = int(entrada) - 1
            if 0 <= seleccion < len(vcf_files):
                return vcf_files[seleccion]
            ConsoleIO.print_warning("Selección fuera de rango.")
        except ValueError:
            ConsoleIO.print_warning("Por favor, introduce un número válido.")


def decodificar_genotipo(record, sample_id: str) -> dict[str, Any]:
    """
    Interpreta dinámicamente el genotipo, manejando multialélicos y nulos.
    """
    # Obtener llamada de genotipo (ej: (0, 1) o (None, None))
    gt_tuple = record.samples[sample_id]["GT"]

    # Caso: Dato faltante (./.)
    if None in gt_tuple:
        return {"tipo": "No Llamado/Missing", "alelos": "./."}

    # Mapear índices a bases
    # record.alleles es una tupla con (REF, ALT1, ALT2...)
    # Ej: record.alleles = ('A', 'T', 'G') -> índice 0='A', 1='T', 2='G'
    alelos_decodificados = [record.alleles[idx] for idx in gt_tuple]
    alelos_str = "/".join(alelos_decodificados)

    # Determinar tipo
    if len(set(gt_tuple)) == 1:
        # Si todos los índices son iguales (0/0, 1/1, 2/2)
        if gt_tuple[0] == 0:
            tipo = "Homocigoto Referencia (WT)"
        else:
            tipo = "Homocigoto Alternativo"
    elif 0 in gt_tuple:
        # Índices distintos (0/1, 1/2)
        tipo = "Heterocigoto"
    else:
        tipo = "Heterocigoto Compuesto (Alt1/Alt2)"

    return {"tipo": tipo, "alelos": alelos_str}


def procesar_paciente(
    vcf_path: Path | str, fasta_path: Path, region: str | None = None
) -> Generator[dict, None, None]:
    """
    Generador que procesa variantes. Usa 'yield' para eficiencia de memoria.
    Permite filtrar por región (ej: 'chr1:1000-2000').
    """
    if not Path(vcf_path).exists():
        raise FileNotFoundError(f"No existe el VCF: {vcf_path}")

    with pysam.FastaFile(str(fasta_path)) as genome, pysam.VariantFile(str(vcf_path)) as vcf:
        sample_id = next(iter(vcf.header.samples))
        print(f"\n🔬 Analizando: {sample_id} | Archivo: {Path(vcf_path).name}")
        print("-" * 60)

        iterator = vcf.fetch(region=region) if region else vcf

        for record in iterator:
            # Saltar variantes sin ALT (bloques homocigotos de referencia típicos en gVCF)
            if not record.alts:
                continue

            # --- VALIDACIÓN DE INTEGRIDAD ---
            # start en pysam es 0-based, stop es exclusivo
            try:
                ref_genome = genome.fetch(record.chrom, record.start, record.stop).upper()
            except KeyError:
                print(f"⚠️ Contig {record.chrom} no encontrado en FASTA.")
                continue

            if ref_genome != record.ref:
                print(
                    f"🚨 MISMATCH {record.chrom}:{record.pos}. VCF_REF={record.ref} vs FASTA={ref_genome}"
                )
                continue

            # --- DECODIFICACIÓN ---
            info_gt = decodificar_genotipo(record, sample_id)

            # Estructurar resultado
            variant_data = {
                "chrom": record.chrom,
                "pos": record.pos,
                "ref": record.ref,
                "alts": record.alts,
                "quality": record.qual,
                "genotype": info_gt["alelos"],
                "zygosity": info_gt["tipo"],
            }

            yield variant_data


if __name__ == "__main__":
    archivo_seleccionado = seleccionar_vcf()

    if archivo_seleccionado:
        # Ejemplo: Procesar y guardar en una lista
        procesador = procesar_paciente(archivo_seleccionado, RUTA_FASTA)

        try:
            for variante in procesador:
                # Aquí puedes filtrar solo lo que te interesa imprimir
                if "Homocigoto Referencia" not in variante["zygosity"]:
                    ConsoleIO.print_dna(
                        f" {variante['chrom']}:{variante['pos']} ({variante['ref']}->{variante['alts']})"
                    )
                    print(f"   └── {variante['zygosity']} [{variante['genotype']}]")
        except Exception as e:
            ConsoleIO.print_error(f"Error durante el procesamiento: {e}")
