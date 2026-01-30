# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
# Licensed under the GNU GPLv3. See LICENSE file in the project root.

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

# --- Imports del Proyecto ---
from src.cfg.manager import DIRS
from src.utils.logger_setup import setup_logging

# --- Setup de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT))

# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def _parse_arguments() -> argparse.Namespace:
    """
    Defines and parses command-line arguments.
    Organized into logical groups for better help output.
    """
    parser = argparse.ArgumentParser(
        description="Pharmagen CLI - Pharmacogenetic Deep Learning Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
    Examples:
    Interactive Mode:
        $ python main.py

    Training (Standard):
        $ python main.py --mode train --model Phenotype_Effect_Outcome --input data/train.tsv --epochs 100

    Training (Optimization):
        $ python main.py --mode train --model Features-Phenotype --input data/train.tsv --optuna --trials 50

    Prediction:
        $ python main.py --mode predict --model Phenotype_Effect_Outcome --input data/new_patients.csv --output results/
            """
    )

    # --- Core Arguments ---
    core_group = parser.add_argument_group("Core Execution")
    core_group.add_argument(
        "--mode", 
        choices=["menu", "train", "predict"], 
        default="menu", 
        help="Execution mode (default: menu)"
    )
    core_group.add_argument("--model", 
        type=str, 
        help="Model name identifier (Required for train/predict modes)"
    )
    core_group.add_argument(
        "--input", 
        type=str, 
        help="Path to input dataset (CSV/TSV)"
    )
    core_group.add_argument(
        "--output", 
        type=str, 
        help="Path to output directory/file (For predictions)"
    )
    core_group.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable debug logging level"
    )

    # --- Training Configuration ---
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument(
        "--epochs", 
        type=int, 
        default=50, 
        help="Number of training epochs (default: 50)"
    )
    train_group.add_argument(
        "--batch-size", 
        type=int, 
        help="Override config batch size"
    )
    train_group.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device (cuda/cpu)"
    )

    # --- Optuna Optimization ---
    opt_group = parser.add_argument_group("Hyperparameter Optimization")
    opt_group.add_argument(
        "--optuna", 
        action="store_true", 
        help="Enable Optuna HPO mode"
    )
    opt_group.add_argument(
        "--trials", 
        type=int, 
        default=20, 
        help="Number of HPO trials to run (default: 20)"
    )

    return parser.parse_args()

# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main():
    args = _parse_arguments()

    # 1. Setup Logging
    setup_logging()
    logger = logging.getLogger("Pharmagen.Main")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")

    try:
        # --- INTERACTIVE MODE ---
        if args.mode == "menu":
            from src.interface.cli import main_menu_loop
            main_menu_loop()

        # --- HEADLESS TRAIN ---
        elif args.mode == "train":
            if not args.model or not args.input:
                parser = argparse.ArgumentParser() # Dummy to print help
                logger.error("Missing required args for training: --model and --input")
                print("\nError: --model and --input are required for 'train' mode.\n")
                sys.exit(1)

            logger.info(f"Starting Training Routine: {args.model}")
            logger.info(f"Device: {args.device} | Epochs: {args.epochs}")

            if args.optuna:
                from src.optuna_tuner import TunerConfig, OptunaOrchestrator
                from src.data_handler import DataConfig
                from src.cfg.manager import get_model_config, MULTI_LABEL_COLS, DIRS
                
                logger.info("Initializing Optuna HPO...")
                
                # 1. Load Config
                raw_cfg = get_model_config(args.model)
                data_dict = raw_cfg["data"]
                search_space = raw_cfg.get("optuna", {})
                
                if not search_space:
                    logger.error(f"Model '{args.model}' has no [optuna] section in models.toml")
                    sys.exit(1)

                # 2. Setup DataConfig
                data_cfg = DataConfig(
                    dataset_path=Path(args.input),
                    feature_cols=data_dict["features"],
                    target_cols=data_dict["targets"],
                    multi_label_cols=list(MULTI_LABEL_COLS),
                    stratify_col=data_dict.get("stratify_col"),
                    num_workers=4
                )

                # 3. Setup TunerConfig
                tuner_cfg = TunerConfig(
                    study_name=f"{args.model}_study",
                    n_trials=args.trials,
                    storage_url=f"sqlite:///{DIRS['reports'] / 'optuna_study.db'}"
                )

                # 4. Run Optimization
                orchestrator = OptunaOrchestrator(
                    tuner_cfg, 
                    data_cfg, 
                    search_space, 
                    device=args.device
                )
                orchestrator.run()
                
            else:
                from src.pipeline import train_pipeline
                train_pipeline(
                    args.model, 
                    Path(args.input), 
                    epochs=args.epochs,
                    batch_size=args.batch_size
                )

        # --- HEADLESS PREDICT ---
        elif args.mode == "predict":
            if not args.model or not args.input:
                logger.error("Missing required args for prediction: --model and --input")
                sys.exit(1)

            logger.info(f"Starting Inference: {args.model}")
            from src.predict import PGenPredictor

            predictor = PGenPredictor(args.model, device=args.device)
            if args.infer == "single":
                results = predictor.predict_single(args.input)
            elif args.infer == "file":
                results = predictor.predict_file(Path(args.input))
            else:
                logger.error("Invalid inference type specified.")
                sys.exit(1)
            if results:
                in_path = Path(args.input)
                if args.output:
                    out_path = Path(args.output)
                    if out_path.is_dir():
                        out_path = out_path / f"{in_path.stem}_preds.tsv"
                else:
                    out_path = PROJECT_ROOT / "reports" / f"{in_path.stem}_preds.tsv"
                
                pd.DataFrame(results).to_csv(out_path, index=False, sep="\t", encoding="utf-8")
                logger.info(f"Predictions saved to {out_path}")
            else:
                logger.warning("No predictions generated (check input file format).")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Critical System Error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
