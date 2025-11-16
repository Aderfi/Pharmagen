from glob import glob
import pysam
from pathlib import Path

import src.config.config as cfg
from src.config.config import REF_GENOME_FASTA, REF_GENOME_FAI

# Rutas a tus archivos
ruta_fasta = REF_GENOME_FASTA
ruta_fai = REF_GENOME_FAI
ruta_vcf = Path(cfg.DATA_DIR, "raw") # Debe estar comprimido y tener su índice .tbi

    




def seleccionar_vcf(vcf_path=ruta_vcf):
    # Listar todos los VCFs disponibles
    vcf_files = glob(str(vcf_path / "*.vcf.gz"))
    if not vcf_files:
        print("No se encontraron archivos VCF en la ruta especificada.")
        return None

    print("Archivos VCF disponibles:")
    for i, vcf_file in enumerate(vcf_files):
        print(f"{i + 1}. {Path(vcf_file).name.strip('.vcf.gz')}")

    seleccion = int(input("Selecciona el número del paciente a analizar: ")) - 1
    if 0 <= seleccion < len(vcf_files):
        return Path(vcf_files[seleccion])
    else:
        print("Selección inválida.")
        return None


def procesar_paciente(vcf_file, fasta_path):
    # 1. Abrir conexiones (Lazy loading)
    genome = pysam.FastaFile(fasta_path)
    vcf = pysam.VariantFile(vcf_file)
    
    # Asumimos que el VCF es de un solo paciente (común en clínica)
    # Si hubiera más, tendrías que iterar por sample.
    sample_id = list(vcf.header.samples)[0]
    #print(f"Analizando paciente: {sample_id}")
    #print("-" * 50)

    # 2. Iterar sobre las variantes del VCF
    for record in vcf:
        
        # Datos básicos del VCF
        chrom = record.chrom
        pos_1based = record.pos  # VCF usa coordenadas base-1
        ref_vcf = record.ref     # Alelo de referencia según el VCF
        alts_vcf = record.alts   # Lista de posibles variantes (ej. ['T'])
        
        if not alts_vcf: continue # Si no hay variantes, saltar
        
        # --- VALIDACIÓN DE SEGURIDAD ---
        # Consultamos el genoma real para ver si el VCF dice la verdad
        # pysam fetch usa base-0, así que restamos 1 al start
        ref_genome = genome.fetch(chrom, pos_1based - 1, int(pos_1based + len(ref_vcf) - 1)).upper() #type: ignore
        
        if ref_genome != ref_vcf:
            print(f"🚨 ERROR CRÍTICO en {chrom}:{pos_1based}. ")
            print(f"   El VCF dice que la ref es {ref_vcf}, pero el genoma dice {ref_genome}.")
            print("   Posible error de versión (hg19 vs hg38). Saltando variante.")
            continue

        # --- INTERPRETACIÓN DEL GENOTIPO ---
        # record.samples[sample_id]['GT'] devuelve una tupla, ej: (0, 1)
        # 0 = Referencia, 1 = Primera Alternativa, 2 = Segunda...
        gt = record.samples[sample_id]['GT']
        
        # Interpretar la tupla
        if gt == (0, 0):
            tipo = "Wild Type (Normal)"
            alelos = f"{ref_vcf}/{ref_vcf}"
        elif gt == (0, 1):
            tipo = "Heterocigoto"
            alelos = f"{ref_vcf}/{alts_vcf[0]}"
        elif gt == (1, 1):
            tipo = "Homocigoto Alternativo"
            alelos = f"{alts_vcf[0]}/{alts_vcf[0]}"
        elif gt == (1, 2):
             tipo = "Heterocigoto Compuesto (Dos variantes distintas)"
             alelos = f"{alts_vcf[0]}/{alts_vcf[1]}"
        else:
            tipo = "Otro/No llamado"
            alelos = "?"

        # Imprimir resultado útil para tu base de datos
        print(f"Gen: {chrom} | Pos: {pos_1based} | Ref: {ref_vcf}")
        print(f"   -> Variante detectada: {alts_vcf}")
        print(f"   -> Genotipo Paciente: {gt} => {alelos} ({tipo})")
        print("-" * 20)

    genome.close()
    vcf.close()




if __name__ == "__main__":
    
    paciente_vcf = seleccionar_vcf()
    
    paciente_vcf = paciente_vcf.name.strip('.vcf.gz') if isinstance(paciente_vcf, Path) else None
    
    if isinstance(paciente_vcf, str):
        procesar_paciente(paciente_vcf, ruta_fasta)
