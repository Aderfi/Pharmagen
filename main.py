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
Pharmagen - Punto de Entrada Principal (CLI & Orquestador).

Este script actúa como la interfaz principal de ejecución para el software Pharmagen.
Su responsabilidad es inicializar el entorno, configurar el sistema de logging global
y enrutar la solicitud del usuario hacia el módulo correspondiente (Entrenamiento,
Predicción o Interfaz Interactiva).

Uso:
    El script puede ejecutarse en dos modalidades:
    1. Interactivo (Por defecto): Lanza un menú visual.
    2. Headless (CLI): Ejecuta tareas específicas mediante argumentos.

Ejemplos:
    # 1. Iniciar menú interactivo
    $ python main.py

    # 2. Entrenar un modelo específico automáticamente
    $ python main.py --mode train --model Phenotype_Effect_Outcome --input data/train.tsv

    # 3. Realizar predicciones sobre un archivo nuevo
    $ python main.py --mode predict --model Phenotype_Effect_Outcome --input data/pacientes.csv

Author:
    Adrim Hamed Outmani (@Aderfi)

Copyright:
    (C) 2025 Adrim Hamed Outmani. Licensed under GNU GPLv3.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# --- Imports del Proyecto ---
from src.cfg.manager import DIRS
from src.utils.logger_setup import setup_logging

# --- Setup de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    # 1. Setup Environment
    setup_logging()
    logger = logging.getLogger("Pharmagen.Main")

    # 2. Argument Parsing
    parser = argparse.ArgumentParser(description="Pharmagen CLI Manager")
    parser.add_argument("--mode", choices=["menu", "train", "predict"], default="menu", help="Execution mode")
    parser.add_argument("--model", type=str, help="Model name (Required for headless modes)")
    parser.add_argument("--input", type=str, help="Input file path (CSV/TSV)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for training")
    parser.add_argument("--optuna", action="store_true", help="Enable Optuna optimization (Train mode only)")
    parser.add_argument("--trials", type=int, default=20, help="Number of Optuna trials")

    args = parser.parse_args()

    try:
        # --- INTERACTIVE MODE ---
        if args.mode == "menu":
            from src.interface.cli import main_menu_loop # noqa
            main_menu_loop()

        # --- HEADLESS TRAIN ---
        elif args.mode == "train":
            if not args.model or not args.input:
                logger.error("--model and --input are required for headless training.")
                sys.exit(1)

            logger.info(f"Starting Headless Training: {args.model}")

            if args.optuna:
                from src.optuna_tuner import run_optuna_study # noqa
                run_optuna_study(args.model, Path(args.input), n_trials=args.trials)
            else:
                from src.pipeline import train_pipeline # noqa
                train_pipeline(args.model, Path(args.input), epochs=args.epochs)

        # --- HEADLESS PREDICT ---
        elif args.mode == "predict":
            if not args.model or not args.input:
                logger.error("--model and --input are required for headless prediction.")
                sys.exit(1)

            logger.info(f"Starting Headless Prediction: {args.model}")
            from src.predict import PGenPredictor # noqa

            predictor = PGenPredictor(args.model)
            results = predictor.predict_file(Path(args.input))

            # Save results
            out_path = Path(args.input).parent / f"{Path(args.input).stem}_preds.csv"
            pd.DataFrame(results).to_csv(out_path, index=False)
            logger.info(f"Predictions saved to {out_path}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical System Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
