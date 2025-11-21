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
#
#!/usr/bin/env python3
# coding=utf-8
"""
Software: pharmagen_pmodel
Versión: 
Autor: Astordna / Aderfi / Adrim Hamed Outmani
Fecha: 2024-06-15
Descripción: Punto de entrada principal del software Pharmagen. 
             Orquesta el flujo entre procesamiento de datos, 
             entrenamiento (Standard/Optuna) y predicción (Inferencia).
"""

import argparse
from datetime import datetime
import sys
import logging
import pandas as pd
from pathlib import Path

# --- Setup de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# --- Imports del Proyecto ---
from src.cfg.config import LOGS_DIR
from src.interface.cli import main_menu_loop
from src.pipeline import train_pipeline
from src.optuna_tuner import run_optuna_study
from src.predict import PGenPredictor

from src.utils.system import check_environment_and_setup
from src.utils.logger import setup_logging

# Constantes
DATE_STAMP = datetime.now().strftime('%Y-%m-%d')


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    check_environment_and_setup()
    
    setup_logging()
    logger = logging.getLogger("Pharmagen")
    
    parser = argparse.ArgumentParser(description="Pharmagen CLI Manager")
    parser.add_argument("--mode", choices=["train", "predict", "menu"], default="menu", help="Modo de ejecución")
    parser.add_argument("--model", type=str, help="Nombre del modelo (para automatización)")
    parser.add_argument("--input", type=str, help="Ruta al archivo de entrada (CSV/TSV)")
    parser.add_argument("--optuna", action="store_true", help="Activar optimización con Optuna (solo modo train)")
    
    args = parser.parse_args()

    try:
        # Modo Interactivo (Por defecto)
        if args.mode == "menu":
            main_menu_loop()
            
        # Modo Entrenamiento (Headless/Automatizado)
        elif args.mode == "train":
            if not args.model or not args.input:
                print("❌ Error: --model y --input son obligatorios en modo 'train'")
                sys.exit(1)
            
            logger.info(f"Iniciando entrenamiento headless: {args.model}")
            if args.optuna:
                run_optuna_study(args.model, args.input)
            else:
                train_pipeline(Path(args.input), args.model)
                
        # Modo Predicción (Headless/Automatizado)
        elif args.mode == "predict":
            if not args.model or not args.input:
                print("❌ Error: --model y --input son obligatorios en modo 'predict'")
                sys.exit(1)
            
            logger.info(f"Iniciando predicción headless: {args.model}")
            predictor = PGenPredictor(args.model)
            results = predictor.predict_file(args.input)
            
            # Guardado automático
            out_name = f"{Path(args.input).stem}_preds_{DATE_STAMP}.csv"
            pd.DataFrame(results).to_csv(out_name, index=False)
            print(f"Predicciones guardadas en: {out_name}")
            
    except KeyboardInterrupt:
        print("\nOperación cancelada por el usuario.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Error no controlado en Main: {e}", exc_info=True)
        print(f"\n❌ Error crítico del sistema: {e}")
        print(f"Consulte el log para más detalles: {LOGS_DIR}")
        sys.exit(1)

if __name__ == "__main__":
    main()