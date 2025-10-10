#   Centralización de la captura de errores del software
#   Todos los errores no controlados se registran en la carpeta 'logs'

import logging
import sys
import os
import datetime
from pathlib import Path
from .config import LOGS_DIR

def unit_logging():    #   Captura de errores del programa
        
    error_date = datetime.datetime.now().strftime("%d-%m-%Y")
    
    log_dir = LOGS_DIR
    if not log_dir.exists():
        os.makedirs(log_dir)
    log_file = log_dir / f"log_{error_date}.txt"

    # Configuración básica del logger
    # Esto captura todos los mensajes (INFO, WARNING, ERROR) y los envía al archivo
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout) # Opcional: para ver logs también en la consola
        ]
    )
    logging.basicConfig(level=logging.INFO)

    # --- La parte clave: Capturar excepciones no controladas ---
    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        Función personalizada que se ejecuta cuando ocurre un error
        que no ha sido capturado por un bloque try...except.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # No captura la interrupción por Ctrl+C para poder salir del programa
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Registra el error en el archivo de log con el traceback completo
        logging.critical("Excepción no controlada:", exc_info=(exc_type, exc_value, exc_traceback))
        
        # Opcional: Muestra un mensaje amigable al usuario en la consola
        print(f"\n❌ Ha ocurrido un error crítico. Revisa el archivo '{log_file}' para más detalles.")

    # Reemplaza el manejador de excepciones por defecto de Python por el nuestro
    sys.excepthook = handle_exception


__all__ = ['unit_logging']