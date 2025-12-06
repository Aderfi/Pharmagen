import pandas as pd
import torch
import os
from tqdm import tqdm # Barra de progreso profesional
from farmagene_grapher import GenomeDataManager, GenomicEmbedder, PharmacogeneGraphBuilder, CHROM_MAPPING
import os

# ==========================================
# CONFIGURACIÓN
# ==========================================
PATH_FASTA = "data/ref_genome/GCF_000001405.40_GRCh38.p14_genomic.fa" # Ajusta tu ruta
PATH_GTF = "data/ref_genome/genomic.gtf"   # Ajusta tu ruta
INPUT_CSV = "BACKUPS/snp_result_cleaned.tsv"                     # Tu archivo de SNPs

OUTPUT_DIR = "F:/g_graph"                 # Donde se guardarán los .pt

# Parámetros del Modelo
MODEL_SIZE = "500m"  # "2.5b" si te sientes valiente con la 4070 Ti Super
CONTEXT_WINDOW = 20  # Bases a cada lado del SNP

# ==========================================
# LÓGICA DE PROCESAMIENTO
# ==========================================

def parse_alleles(variant_str):
    """
    Convierte 'T>A,C' en una lista de alelos alternativos ['A', 'C'].
    Maneja casos complejos.
    """
    if '>' not in variant_str:
        return [] # Casos raros o indels mal formados
    
    _, alts = variant_str.split('>')
    return alts.split(',')

def find_gene_coordinates(gene_name, gtf_df):
    """
    Busca las coordenadas de un gen en el DataFrame del GTF cargado.
    Maneja genes múltiples tipo 'CYP21A2;TNXB' probando uno por uno.
    """
    candidates = gene_name.split(';')
    
    for gene in candidates:
        # Buscar coincidencia exacta en la columna gene_name que creamos
        gene_rows = gtf_df[gtf_df['gene_name'] == gene]
        
        if not gene_rows.empty:
            # Encontramos el gen. Devolvemos cromosoma, inicio min y fin max.
            chrom = gene_rows.iloc[0]['seqname']
            start = gene_rows['start'].min()
            end = gene_rows['end'].max()
            return gene, chrom, start, end
            
    return None, None, None, None

def main():
    # 1. Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 2. Cargar Datos de Entrada
    print(f"--- Leyendo archivo de SNPs: {INPUT_CSV} ---")
    # sep='\s+' maneja espacios variables o tabuladores automáticamente
    df_snps = pd.read_csv(INPUT_CSV, sep='\t', header=0, dtype={'snp_id': str, 'chr': str, 'pos': int, 'variant': str, 'variant_type': str, 'gene': str})
    print(f"Total SNPs a procesar: {len(df_snps)}")

    # 3. Inicializar Motores (Esto carga la VRAM)
    print("\n--- Inicializando Motores Genómicos ---")
    data_manager = GenomeDataManager(PATH_FASTA, PATH_GTF)
    embedder = GenomicEmbedder(model_size=MODEL_SIZE)
    graph_builder = PharmacogeneGraphBuilder(embedder, context_window=CONTEXT_WINDOW)

    # 4. Agrupar por Gen para optimizar lecturas de disco
    # En lugar de saltar aleatoriamente, procesamos gen por gen.
    grouped_snps = df_snps.groupby('gene')
    
    successful_graphs = 0
    errors = []

    print("\n--- Iniciando Bucle de Generación ---")
    
    for raw_gene_name, group in tqdm(grouped_snps, desc="Procesando Genes"):
        
        # A. Localizar Gen en GTF
        real_gene_name, chrom, start, end = find_gene_coordinates(raw_gene_name, data_manager.gtf)
        
        if real_gene_name is None:
            errors.append(f"Gen no encontrado en GTF: {raw_gene_name}")
            continue

        # B. Extraer Secuencia Base (Una vez por gen)
        try:
            # Convertimos cromosomas tipo '11' a 'chr11' si es necesario para el filtrado GTF
            # o usamos la lógica interna del manager.
            # Asumimos que el GTF ya tiene el formato correcto en 'chrom'
            
            ref_seq, anno_mask = data_manager.extract_gene_data(
                chrom, start, end, target_gene=real_gene_name
            )
        except Exception as e:
            errors.append(f"Error extrayendo {real_gene_name}: {str(e)}")
            continue

        # C. Procesar SNPs dentro de este Gen
        for _, row in group.iterrows():
            snp_id = row['snp_id']
            genomic_pos = row['pos'] # Posición absoluta en cromosoma
            variant_str = row['variant']
            
            # Calcular posición relativa dentro de la secuencia extraída
            # start es base-1, genomic_pos es base-1.
            # En Python (base-0): pos_relativa = genomic_pos - start
            relative_pos = genomic_pos - start
            
            # Validación de límites
            if relative_pos < 0 or relative_pos >= len(ref_seq):
                errors.append(f"SNP {snp_id} fuera de rangos del gen {real_gene_name}")
                continue

            # Obtener alelos (puede haber múltiples, ej. A,C,G)
            alt_alleles = parse_alleles(variant_str)
            
            for alt in alt_alleles:
                try:
                    # Crear lista de variantes para el Builder
                    # El builder espera lista de dicts
                    variant_data = [{
                        'pos': relative_pos,
                        'alt': alt,
                        'type': row['variant_type']
                    }]
                    
                    # --- CONSTRUIR GRAFO ---
                    graph = graph_builder.build(ref_seq, variant_data, anno_mask)
                    
                    # Añadir metadatos útiles al objeto Data antes de guardar
                    graph.snp_id = snp_id
                    graph.gene = real_gene_name
                    graph.clin_sig = row['clin_sig']
                    graph.chromosome = chrom
                    graph.position = genomic_pos
                    
                    # --- GUARDAR ---
                    # Nombre archivo: GEN_rsID_Alelo.pt
                    filename = f"{real_gene_name}_rs{snp_id}_{alt}.pt"
                    save_path = os.path.join(OUTPUT_DIR, filename)
                    torch.save(graph, save_path)
                    
                    successful_graphs += 1
                    
                except Exception as e:
                    errors.append(f"Fallo al construir grafo {snp_id} ({alt}): {str(e)}")

    # 5. Reporte Final
    print("\n" + "="*40)
    print(f"PROCESO TERMINADO")
    print(f"Grafos generados exitosamente: {successful_graphs}")
    print(f"Errores encontrados: {len(errors)}")
    print("="*40)
    
    if errors:
        print("Primeros 5 errores:")
        for err in errors[:5]:
            print(f"- {err}")
        print(f"(Ver 'error_log.txt' para detalles completos)")
        with open("error_log.txt", "w") as f:
            f.write("\n".join(errors))

if __name__ == "__main__":
    main()