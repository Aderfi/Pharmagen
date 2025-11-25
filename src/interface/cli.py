# Pharmagen - Command Line Interface
# Interactive Menu and Workflows

import sys
import logging
import datetime
import time
from pathlib import Path

import pandas as pd

# Project Imports
from src.cfg.manager import get_available_models, DIRS, PROJECT_ROOT
from src.interface.utils import Spinner, input_path, print_error, print_header, print_success
from src.optuna_tuner import run_optuna_study
from src.pipeline import train_pipeline
from src.predict import PGenPredictor
from src.utils.io import print_conditions_details, print_gnu_notice, print_warranty_details

logger = logging.getLogger(__name__)
DATE_STAMP = datetime.datetime.now().strftime("%Y_%m_%d")

# ==============================================================================
# UTILS
# ==============================================================================

def _select_model() -> str:
    """Interactively select a model from configuration."""
    models = get_available_models()
    if not models:
        print_error("No models found in models.toml")
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
        print_error("Invalid selection.")

# ==============================================================================
# WORKFLOWS
# ==============================================================================

def run_genomic_processing():
    """Simulation of Genomic ETL."""
    print_header("Genomic Processing Module")
    logger.info("Starting interactive genomic module.")
    
    with Spinner("Analyzing VCF files..."):
       time.sleep(2) 
        
    print_success("Processing completed (Simulated).")

def run_training_flow():
    """Interactive Training Workflow."""
    print_header("Training Module")
    
    # 1. Select Model
    model_name = _select_model()
    
    # 2. Select Data
    default_data = DIRS["data"] / "processed" / "train_data.tsv"
    if not default_data.exists():
        # Fallback to project root default if exists
        fallback = PROJECT_ROOT / "train_data" / "final_enriched_data.tsv"
        if fallback.exists():
            default_data = fallback
        else:
            default_data = None
    
    csv_path = input_path("Training Data Path (CSV/TSV)", default=default_data)

    # 3. Select Mode
    print("\nTraining Mode:")
    print("  1. Standard Training (Single Run)")
    print("  2. Hyperparameter Optimization (Optuna)")
    
    mode = input("Select (1-2): ").strip()
    
    if mode == "1":
        _run_standard_training(model_name, csv_path)
    elif mode == "2":
        _run_optuna_training(model_name, csv_path)
    else:
        print_error("Invalid option.")

def _run_standard_training(model_name: str, csv_path: Path):
    epochs_str = input("Epochs [default -> 100]: ").strip() 
    epochs = int(epochs_str) if epochs_str.isdigit() else 100
    
    print(f"\nStarting Standard Training: '{model_name}'")
    train_pipeline(model_name=model_name, csv_path=str(csv_path), epochs=epochs)
    print_success("Training finished.")

def _run_optuna_training(model_name: str, csv_path: Path):
    trials_str = input("Number of Trials [default -> 50]: ").strip()
    n_trials = int(trials_str) if trials_str.isdigit() else 50
    
    print(f"\nStarting Optuna Optimization: '{model_name}' ({n_trials} trials)")
    run_optuna_study(model_name=model_name, csv_path=csv_path, n_trials=n_trials)
    print_success("Optimization finished.")

def run_prediction_flow():
    """Interactive Prediction Workflow."""
    print_header("Prediction Module")
    
    model_name = _select_model()
    
    try:
        with Spinner("Loading model..."):
            predictor = PGenPredictor(model_name)
        print_success("Model loaded.")

        while True:
            print("\n--- Prediction Menu ---")
            print("  1. Interactive (Single)")
            print("  2. Batch (File)")
            print("  3. Back to Main Menu")
            
            sub_choice = input("Option: ").strip()

            if sub_choice == "1":
                _interactive_predict_loop(predictor)
            elif sub_choice == "2":
                _batch_predict_flow(predictor)
            elif sub_choice == "3":
                break
                
    except FileNotFoundError as e:
        logger.error(f"Model loading failed: {e}")
        print_error(f"Could not load model: {e}")
        print("Tip: Train the model first.")
    except Exception as e:
        logger.error(f"Critical prediction error: {e}", exc_info=True)
        print_error(f"Unexpected error: {e}")

def _interactive_predict_loop(predictor: PGenPredictor):
    print("\n(Type 'q' to cancel)")
    inputs = {}
    
    for feature in predictor.feature_cols:
        val = input(f"Value for '{feature}': ").strip()
        if val.lower() == 'q': return
        inputs[feature] = val
    
    print("\nCalculating...")
    result = predictor.predict_single(inputs)
    
    print("\n--- Result ---")
    if result:
        for k, v in result.items():
            print(f"  🔹 {k}: {v}")
    else:
        print_error("Prediction failed.")

def _batch_predict_flow(predictor: PGenPredictor):
    path = input_path("Input CSV/TSV Path")
    
    with Spinner(f"Processing {path.name}..."):
        results = predictor.predict_file(path)
    
    if not results:
        print("⚠️ No results generated.")
        return

    out_path = path.parent / f"{path.stem}_predictions_{DATE_STAMP}.csv"
    pd.DataFrame(results).to_csv(out_path, index=False)
    print_success(f"Predictions saved to: {out_path}")

def run_advanced_analysis():
    print_header("Advanced Analysis")
    print("Generating interpretability reports...")
    print("(Under Construction)")

# ==============================================================================
# MAIN MENU LOOP
# ==============================================================================

def main_menu_loop():
    logger.info("Starting interactive menu.")
    print_gnu_notice()
    
    while True:
        print_header(f"Pharmagen - Main Menu")
        print("  1. Genomic Processing (ETL)")
        print("  2. Train Models (Deep Learning)")
        print("  3. Predict (Inference)")
        print("  4. Advanced Analysis")
        print("  5. Exit")
        print("="*60)
        
        choice = input("Select option (1-5): ").strip()
        
        if choice == "show w":
            print_warranty_details()
            continue
            
        if choice == "show c":
            print_conditions_details()
            continue

        if choice == "1":
            run_genomic_processing()
        elif choice == "2":
            run_training_flow()
        elif choice == "3":
            run_prediction_flow()
        elif choice == "4":
            run_advanced_analysis()
        elif choice == "5":
            logger.info("User exit.")
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main_menu_loop()
