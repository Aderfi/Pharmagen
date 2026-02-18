# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy via Deep Learning
# Copyright (C) 2025  Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

import datetime
import logging
from pathlib import Path
import sys

from src.cfg.manager import DIRS, METADATA, get_available_models
from src.interface.io import (
    print_conditions_details,
    print_gnu_notice,
    print_warranty_details,
)
from src.interface.ui import ConsoleIO, Spinner
from src.pipeline import train_pipeline

logger = logging.getLogger(__name__)
DATE_STAMP = datetime.datetime.now().strftime("%Y_%m_%d")


# ==============================================================================
# HELPERS
# ==============================================================================


def _select_model() -> str:
    """Interactively select a model from configuration."""
    models = get_available_models()
    if not models:
        ConsoleIO.print_error("No models found in configuration.")
        sys.exit(1)

    print("\nAvailable Models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m}")

    while True:
        choice = input("\nSelect model (number): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
        ConsoleIO.print_error("Invalid selection.")


# ==============================================================================
# WORKFLOWS
# ==============================================================================


def _run_advanced_analysis():
    """Advanced analysis workflow (placeholder)."""
    ConsoleIO.print_header("Advanced Analysis")
    ConsoleIO.print_warning("NOT IMPLEMENTED YET")
    ConsoleIO.print_info("This module will provide:")
    print("  • Model interpretability reports")
    print("  • Feature importance analysis")
    print("  • SHAP value visualizations")
    print("  • Performance metrics dashboard")


def _run_genomic_processing():
    """NGS Genomic Data Processing Workflow"""
    ConsoleIO.print_header("Pharmagen NGS Processing Module")
    ConsoleIO.print_info("This module will provide:")
    with Spinner("Running NGS Processing Pipeline..."):
        from src.analysis.ngs_pipeline import run_ngs_pipeline
    r1 = ConsoleIO.input_path(
        "Input R1 Path", must_exist=True, file_extensions=[".fastq", ".fq", ".fastq.gz", ".fq.gz"]
    )
    r2 = ConsoleIO.input_path(
        "Input R2 Path", must_exist=True, file_extensions=[".fastq", ".fq", ".fastq.gz", ".fq.gz"]
    )
    sample_name = input("Sample Name: ").strip()
    threads_str = input("Number of Threads [default -> 4]: ").strip()
    threads = int(threads_str) if threads_str.isdigit() else 4
    run_ngs_pipeline(r1=r1, r2=r2, sample_name=sample_name, threads=threads)


def _run_training_flow():
    """Interactive Training Workflow."""
    ConsoleIO.print_header("Training Module")

    # 1. Select Model
    model_name = _select_model()

    # 2. Select Data
    default_data = DIRS["data"] / "processed" / "train_data.tsv"
    if not default_data.exists():
        default_data = None

    csv_path = ConsoleIO.input_path("Training Data Path (CSV/TSV)", default=default_data)

    # 3. Select Mode
    print("\nTraining Mode:")
    print("  1. Standard Training (Single Run)")
    print("  2. Hyperparameter Optimization (Optuna)")

    mode = ConsoleIO.input_choice("Select (1-2)", ["1", "2"])

    if mode == "1":
        _run_standard_training(model_name, csv_path)
    elif mode == "2":
        _run_optuna_training(model_name, csv_path)


def _run_standard_training(model_name: str, csv_path: Path):
    epochs_str = input("Epochs [default -> 50]: ").strip()
    epochs = int(epochs_str) if epochs_str.isdigit() else 50

    ConsoleIO.print_step(f"Starting Standard Training: '{model_name}'")

    # Calls the Refactored Pipeline
    try:
        train_pipeline(model_name=model_name, csv_path=csv_path, epochs=epochs)
        ConsoleIO.print_success("Training Pipeline Completed Successfully.")
    except Exception as e:
        logger.exception("Training failed: %s", e)
        ConsoleIO.print_error(f"Training failed. Check logs. Error: {e}")


def _run_optuna_training(model_name: str, csv_path: Path):

    from model.engine.optuna_tuner import OptunaOrchestrator, TunerConfig
    from src.cfg.manager import MULTI_LABEL_COLS, get_model_config
    from src.data.data_handler import DataConfig

    trials_str = input("Number of Trials [default -> 20]: ").strip()
    n_trials = int(trials_str) if trials_str.isdigit() else 20

    ConsoleIO.print_step(f"Starting Optuna Optimization: '{model_name}' ({n_trials} trials)")

    try:
        # 1. Fetch Configuration
        raw_cfg = get_model_config(model_name)
        data_dict = raw_cfg["data"]
        search_space = raw_cfg.get("optuna", {})

        if not search_space:
            ConsoleIO.print_error(f"No [optuna] configuration found for {model_name}.")
            return

        # 2. Build Config Objects
        data_cfg = DataConfig(
            dataset_path=csv_path,
            feature_cols=data_dict["features"],
            target_cols=data_dict["targets"],
            multi_label_cols=list(MULTI_LABEL_COLS),
            stratify_col=data_dict.get("stratify_col"),
            num_workers=4,
        )

        tuner_cfg = TunerConfig(
            study_name=f"{model_name}_interactive",
            n_trials=n_trials,
            storage_url=f"sqlite:///{DIRS['reports'] / 'optuna_study.db'}",
        )

        # 3. Launch Orchestrator
        orchestrator = OptunaOrchestrator(tuner_cfg, data_cfg, search_space)
        study = orchestrator.run()

        ConsoleIO.print_success(f"Optimization finished. Best Value: {study.best_value:.4f}")
        ConsoleIO.print_info(f"Best Params: {study.best_params}")

    except Exception as e:
        logger.exception("Optimization failed: %s", e)
        ConsoleIO.print_error(f"Optimization failed: {e}")


def _run_prediction_flow():
    """Interactive Prediction Workflow."""
    ConsoleIO.print_header("Prediction Module")

    model_name = _select_model()

    try:
        from model.engine.predict import PGenPredictor

        with Spinner("Loading model into memory..."):
            predictor = PGenPredictor(model_name)
        ConsoleIO.print_success("Model loaded.")

        while True:
            print("\n--- Prediction Menu ---")
            print("  1. Interactive (Single Sample)")
            print("  2. Batch (File)")
            print("  3. Back to Main Menu")

            sub_choice = input("Option: ").strip()

            if sub_choice == "1":
                _interactive_predict_loop(predictor)
            elif sub_choice == "2":
                _batch_predict_flow(predictor)
            elif sub_choice == "3":
                break
            else:
                ConsoleIO.print_warning("Invalid option.")

    except ImportError:
        ConsoleIO.print_error("Prediction module (src.predict) is missing or not updated.")
    except FileNotFoundError as e:
        ConsoleIO.print_error(f"Model artifacts not found: {e}")
        ConsoleIO.print_info("Tip: Run Training first.")
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        ConsoleIO.print_error(f"Unexpected error: {e}")


def _interactive_predict_loop(predictor):
    print("\n(Type 'q' to cancel)")
    inputs = {}

    features = getattr(predictor, "feature_cols", ["drug_id", "gene_id"])

    for feature in features:
        val = input(f"Value for '{feature}': ").strip()
        if val.lower() == "q":
            return
        inputs[feature] = val

    ConsoleIO.print_step("Calculating probabilities...")
    try:
        result = predictor.predict_single(inputs)
        print("\n--- Result ---")
        if result:
            for k, v in result.items():
                val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"  🔹 {k}: {val_str}")
    except Exception as e:
        ConsoleIO.print_error(f"Prediction logic failed: {e}")


def _batch_predict_flow(predictor):
    path = ConsoleIO.input_path("Input CSV/TSV Path")

    with Spinner(f"Processing {path.name}..."):
        results = predictor.predict_file(path)

    if results is None:
        return  # Error already handled in predict_file likely

    out_name = f"{path.stem}_preds_{DATE_STAMP}.csv"
    out_path = path.parent / out_name

    # Save results
    import pandas as pd

    pd.DataFrame(results).to_csv(out_path, index=False)
    ConsoleIO.print_success(f"Predictions saved to: {out_path}")


# ==============================================================================
# MAIN MENU LOOP
# ==============================================================================


def main_menu_loop():
    """Main interactive menu loop."""
    logger.info("Starting interactive menu")
    print_gnu_notice()
    ConsoleIO.print_header("Pharmagen - Main Menu")
    ConsoleIO.print_info(f" Version: {METADATA.get('version', 'Not Available')}", metadata=True)
    ConsoleIO.print_info(f" Authors: {METADATA.get('authors', 'Not Available')}", metadata=True)
    ConsoleIO.print_divider("=")
    while True:
        print("  1. Genomic Processing (ETL)")
        print("  2. Train Models (Deep Learning)")
        print("  3. Predict (Inference)")
        print("  4. Advanced Analysis")
        print("  5. Exit")
        print()
        print("  Type 'show w' for warranty details")
        print("  Type 'show c' for license conditions")
        ConsoleIO.print_divider("=")

        choice = input("Select option (1-5): ").strip()

        # Easter eggs for license info
        if choice == "show w":
            print_warranty_details()
            continue
        if choice == "show c":
            print_conditions_details()
            continue

        # Main menu options
        if choice == "1":
            _run_genomic_processing()
        elif choice == "2":
            _run_training_flow()
        elif choice == "3":
            _run_prediction_flow()
        elif choice == "4":
            _run_advanced_analysis()
        elif choice == "5":
            if ConsoleIO.confirm("Are you sure you want to exit?", default=False):
                logger.info("User exit")
                ConsoleIO.print_info("Goodbye!  👋")
                sys.exit(0)
        else:
            ConsoleIO.print_error("Invalid option - please select 1-5")


if __name__ == "__main__":
    main_menu_loop()
