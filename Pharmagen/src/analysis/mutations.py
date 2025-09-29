from Pharmagen.src.data_handle.io import leer_genoma_fasta
from Pharmagen.src.data_handling.processing import limpiar_secuencia
from Pharmagen.src.analysis.mutations import encontrar_mutaciones_comunes

def main():
    """
    Script principal para ejecutar el flujo de búsqueda de mutaciones.
    """
    # 1. Cargar los datos
    genoma = leer_genoma_fasta("../data/raw/gene.fasta")
    
    # 2. Procesar los datos
    secuencia_limpia = limpiar_secuencia(genoma)
    
    # 3. Realizar el análisis
    resultados = encontrar_mutaciones_comunes(secuencia_limpia)
    
    # 4. Guardar o mostrar resultados
    print("Mutaciones encontradas:", resultados)

if __name__ == "__main__":
    main()