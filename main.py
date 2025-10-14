"""
Software: pharmagen_pmodel
Versión: 0.1
Autor: Astordna / Aderfi / Adrim Hamed Outmani
Fecha: 2024-06-15
Descripción: Punto de entrada principal del software. Orquesta todas las funcionalidades,
             incluyendo el tratamiento de datos genómicos, entrenamiento, predicción y análisis.
"""

import itertools, logging, sys, subprocess, threading, time

# Aquí puedes importar funciones generales, utilidades y, si quieres, los pipelines de pgen_model
# from src.data_utils import process_genomic_data
# from src.visualization import visualize_results
# from pgen_model import main as pgen_model_main

logging.basicConfig(level=logging.INFO)

def process_genomic_data():
    print(">> Procesando datos genómicos (pendiente de implementar)")

def advanced_analysis():
    print(">> Análisis avanzado (pendiente de implementar)")

def launch_pgen_model():
    # Llama al menú específico del paquete pgen_model():
        subprocess.run([sys.executable, "-m", "pgen_model"])
    

def loading_animation():
    spinner = itertools.cycle(['|', '/', '-', '\\'])
    for _ in range(20):
        sys.stdout.write(next(spinner))  # write the next character
        sys.stdout.flush()                # flush stdout buffer (actual character display)
        time.sleep(0.1)
        sys.stdout.write('\b')            # erase the last written char

def main():
    print("""
    ================= Pharmagen: MENÚ PRINCIPAL =================
    1. Procesar datos genómicos de entrada
    2. Entrenar/predir modelo (menú modelo ML)
    3. Análisis avanzado (En progreso...)
    4. Salir
    =============================================================""")
    while True:
        choice = input("Selecciona opción (1-4): ").strip()
        if choice == "1":
            process_genomic_data()
        elif choice == "2":
            print("Inicializando el módulo PGen-Model...")
            loading_animation()
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