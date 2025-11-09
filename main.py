"""
Software: pharmagen_pmodel
Versión: 0.1
Autor: Astordna / Aderfi / Adrim Hamed Outmani
Fecha: 2024-06-15
Descripción: Punto de entrada principal del software. Orquesta todas las funcionalidades,
             incluyendo el tratamiento de datos genómicos, entrenamiento, predicción y análisis.
"""

import datetime, itertools, logging, sys, subprocess, threading, time
from src.config.config import LOGS_DIR

# Aquí puedes importar funciones generales, utilidades y, si quieres, los pipelines de pgen_model
# from src.data_utils import process_genomic_data
# from src.visualization import visualize_results
# from pgen_model import main as pgen_model_main



def process_genomic_data(): #1
    logger = logging.getLogger(__name__)
    logger.info("Iniciando procesamiento de datos genómicos")
    print(">> Procesando datos genómicos (pendiente de implementar)")


def advanced_analysis(): #3
    logger = logging.getLogger(__name__)
    logger.info("Iniciando análisis avanzado")
    print(">> Análisis avanzado (pendiente de implementar)")


def launch_pgen_model(): # 2   
    logger = logging.getLogger(__name__)
    logger.info("Lanzando pgen_model")
    # Llama al menú específico del paquete pgen_model():
    subprocess.run([sys.executable, "-m", "pgen_model"])
    time.sleep(2)  # Pausa para asegurar que el subproceso se inicie correctamente
    main()

def help_menu():    #4
    print(">> Menú de ayuda (pendiente de implementar)")
    main()

def loading_animation():
    '''
    #spinner = itertools.cycle(["|", "/", "-", "\\"])
    spinner = itertools.cycle(['["=     ]', '[~<    ]', '[ ~<   ]', '[~~=   ]', '[~~~<  ]', '[ ~~~= ]', '[  ~~~<]', '[   ~~~]', '[    ~~]', '[     ~]', '[      ]'])
    for _ in range(20):
        sys.stdout.write(next(spinner))  # write the next character
        sys.stdout.flush()  # flush stdout buffer (actual character display)
        time.sleep(0.4)
        sys.stdout.write("\b")  # erase the last written char
    '''
    frames = [
        '=     ',
        '~<    ',
        '~~=   ',
        '~~~<  ',
        ' ~~~= ',
        '  ~~~<',
        '   ~~~',
        '    ~~',
        '     ~',
        '      '
    ]
    interval = 0.4
    spinner = itertools.cycle(frames)
    for _ in range(20):
        frame = next(spinner)
        sys.stdout.write(f"\r{frame}") 
        sys.stdout.flush() 
        time.sleep(interval)
    sys.stdout.write("\n")
    sys.stdout.flush()

def simple_dna_animation():
    frames = [
        [
            "    | |    ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  A---T   ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  G---C   ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  T---A   "
        ],
        [
            "   A---T   ",
            "    | |    ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  G---C   ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  T---A   ",
            "   \\ /   ",
            "    X     ",
            "   / \\   "
        ],
        [
            "  G---C   ",
            "   A---T   ",
            "    | |    ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  T---A   ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  A---T   ",
            "   \\ /   ",
            "    X     "
        ],
        [
            "  T---A   ",
            "  G---C   ",
            "   A---T   ",
            "    | |    ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  A---T   ",
            "   \\ /   ",
            "    X     ",
            "   / \\   ",
            "  G---C   ",
            "   \\ /   "
        ]
    ]
    
    interval = 0.3
    spinner = itertools.cycle(frames)
    
    print("Iniciando análisis de ADN...")
    time.sleep(1)
    
    for _ in range(40):
        frame = next(spinner)
        
        # Limpiar y mostrar
        sys.stdout.write("\033[2J\033[H")
        print("🧬 ANÁLISIS DE ADN EN PROGRESO 🧬\n")
        for line in frame:
            print(f"    {line}")
        sys.stdout.flush()
        time.sleep(interval)
    
    sys.stdout.write("\033[2J\033[H")
    print("🧬 Análisis de ADN completado exitosamente! ✅")

def dna_loading_animation():
    frames = [
        [
            "    ╭─╮    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
            "╱         ╲",
            "╲         ╱",
            " ╲       ╱ ",
            "  ╲     ╱  ",
            "   ╲   ╱   ",
            "    ╲ ╱    ",
            "     ╳     ",
            "    ╱ ╲    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
        ],
        [
            "     │     ",
            "    ╭─╮    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
            "╱         ╲",
            "╲         ╱",
            " ╲       ╱ ",
            "  ╲     ╱  ",
            "   ╲   ╱   ",
            "    ╲ ╱    ",
            "     ╳     ",
            "    ╱ ╲    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
        ],
        [
            "    ═══    ",
            "     │     ",
            "    ╭─╮    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
            "╱         ╲",
            "╲         ╱",
            " ╲       ╱ ",
            "  ╲     ╱  ",
            "   ╲   ╱   ",
            "    ╲ ╱    ",
            "     ╳     ",
            "    ╱ ╲    ",
            "   ╱   ╲   ",
        ],
        [
            "   ○─○─○   ",
            "    ═══    ",
            "     │     ",
            "    ╭─╮    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
            "╱         ╲",
            "╲         ╱",
            " ╲       ╱ ",
            "  ╲     ╱  ",
            "   ╲   ╱   ",
            "    ╲ ╱    ",
            "     ╳     ",
            "    ╱ ╲    ",
        ],
        [
            "  ○─────○  ",
            "   ○─○─○   ",
            "    ═══    ",
            "     │     ",
            "    ╭─╮    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
            "╱         ╲",
            "╲         ╱",
            " ╲       ╱ ",
            "  ╲     ╱  ",
            "   ╲   ╱   ",
            "    ╲ ╱    ",
            "     ╳     ",
        ],
        [
            " ○───────○ ",
            "  ○─────○  ",
            "   ○─○─○   ",
            "    ═══    ",
            "     │     ",
            "    ╭─╮    ",
            "   ╱   ╲   ",
            "  ╱     ╲  ",
            " ╱       ╲ ",
            "╱         ╲",
            "╲         ╱",
            " ╲       ╱ ",
            "  ╲     ╱  ",
            "   ╲   ╱   ",
            "    ╲ ╱    ",
        ]
    ]
    
    interval = 0.5
    spinner = itertools.cycle(frames)
    
    for _ in range(30):  # Número de ciclos de animación
        frame = next(spinner)
        
        # Limpiar pantalla y mostrar frame
        sys.stdout.write("\033[2J\033[H")  # Limpiar pantalla
        for line in frame:
            sys.stdout.write(line + "\n")
        sys.stdout.flush()
        time.sleep(interval)
    
    # Limpiar al final
    sys.stdout.write("\033[2J\033[H")
    print("Análisis de ADN completado ✓")

def main():
    
    dt_exec = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logging.basicConfig(
        filename=f"{LOGS_DIR}/Pharmagen_{dt_exec}.log", 
        filemode="w", 
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info("Pharmagen_Log iniciado correctamente")
    
    print(
        """
                                                                ================= \033[1mPharmagen: MENÚ PRINCIPAL\033[0m =================
                                                                
                                                                    1. Procesar datos genómicos
                                                                                    
                                                                    2. Entrenar/Predecir (menú modelo ML)
                                                                                    
                                                                    3. Análisis avanzado (En progreso...)
                                                                    
                                                                    4. Instrucciones y ayuda (En progreso...)
                                                                                    
                                                                            5. Salir
                                                                                    
                                                                ==============================================================
                                                                
    """
    )
    while True:
        try:
            choice = input("Selecciona opción (1-4): ").strip()
            logger.info(f"Usuario seleccionó opción: {choice}")
            if choice == "1":
                process_genomic_data()
            elif choice == "2":
                print("Inicializando el módulo PGen-Model...")
                loading_animation()
                launch_pgen_model()
            elif choice == "3":
                advanced_analysis()
            elif choice == "4":
                help_menu()
            elif choice == "5":
                logger.info("Usuario salió del programa")
                print("¡Gracias por usar Pharmagen!")
                sys.exit(0)
            else:
                logger.warning(f"Opción no válida seleccionada: {choice}")
                print("Opción no válida. Intente de nuevo.")
        except KeyboardInterrupt:
            logger.info("Programa interrumpido por el usuario (Ctrl+C)")
            print("\n¡Gracias por usar Pharmagen!")
            sys.exit(0)
        except Exception as e:
            logger.error(f"Error inesperado en el menú principal: {str(e)}")
            print(f"Ha ocurrido un error: {str(e)}")


if __name__ == "__main__":
    main()
    