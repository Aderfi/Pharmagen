"""
Software: pharmagen_pmodel
Versión: 0.1
Autor: Astordna / Aderfi / Adrim Hamed Outmani
Fecha: 2024-06-15
Descripción: Punto de entrada principal del software. Orquesta todas las funcionalidades,
             incluyendo el tratamiento de datos genómicos, entrenamiento, predicción y análisis.
"""

import sys
import logging

# Aquí puedes importar funciones generales, utilidades y, si quieres, los pipelines de pgen_model
# Ejemplo (ajusta según tus módulos):
# from src.data_utils import process_genomic_data
# from src.visualization import visualize_results
# from pgen_model import main as pgen_model_main

logging.basicConfig(level=logging.INFO)

def process_genomic_data():
    print(">> Procesando datos genómicos (pendiente de implementar)")

def advanced_analysis():
    print(">> Análisis avanzado (pendiente de implementar)")

def launch_pgen_model():
    # Llama al menú específico del paquete pgen_model
    import subprocess
    subprocess.run([sys.executable, "-m", "pgen_model"])

def main():
    print("""
    ================= Pharmagen: MENÚ PRINCIPAL =================
    1. Procesar datos genómicos de entrada
    2. Entrenar/predir modelo (menú modelo ML)
    3. Análisis avanzado (futuro)
    4. Salir
    =============================================================
    """)
    while True:
        choice = input("Selecciona opción (1-4): ").strip()
        if choice == "1":
            process_genomic_data()
        elif choice == "2":
            launch_pgen_model()
        elif choice == "3":
            advanced_analysis()
        elif choice == "4":
            print("¡Gracias por usar Pharmagen!")
            sys.exit(0)
        else:
            print("Opción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()