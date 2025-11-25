# Pharmagen - Optuna Hyperparameter Optimization
#
# Implements a comprehensive hyperparameter optimization pipeline using Optuna.
# Supports multi-objective optimization and clean architecture.

import logging
import json
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Project Imports
from src.cfg.manager import get_model_config, MULTI_LABEL_COLS, DIRS
from src.data_handler import load_dataset, PGenProcessor, PGenDataset
from src.modeling import create_model
from src.losses import MultiTaskUncertaintyLoss
from src.trainer import PGenTrainer

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

class OptunaTuner:
    """
    Orchestrator class for hyperparameter optimization.
    Encapsulates data, configuration, and Optuna lifecycle.
    """

    def __init__(
        self,
        model_name: str,
        csv_path: Union[str, Path],
        n_trials: int = 100,
        epochs: int = 75,
        patience: int = 15,
        random_seed: int = 711,
        use_multi_objective: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.model_name = model_name
        self.csv_path = Path(csv_path)
        self.n_trials = n_trials
        self.epochs = epochs
        self.patience = patience
        self.seed = random_seed
        self.use_multi_objective = use_multi_objective
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model Configuration
        self.config = get_model_config(model_name)
        self.feature_cols = [c.lower() for c in self.config["features"]]
        self.target_cols = [t.lower() for t in self.config["targets"]]
        
        # Validate Optuna space
        self.optuna_space: Dict[str, List[Any]] = self.config.get("params_optuna", {})
        if not self.optuna_space:
            raise ValueError(f"No Optuna search space defined for model '{model_name}'")

        # Deferred Data Initialization
        self.train_dataset: Optional[PGenDataset] = None
        self.val_dataset: Optional[PGenDataset] = None
        self.encoder_dims: Dict[str, int] = {}

        # Prepare data immediately
        self._prepare_data()

    def _prepare_data(self):
        """Loads, processes, and caches datasets in memory (One-time setup)."""
        logger.info(f"Preparing data for {self.model_name}...")
        
        # Define necessary columns
        cols_to_load = list(set(self.feature_cols + self.target_cols))
        
        # 1. Load & Clean
        df = load_dataset(
            csv_path=self.csv_path,
            cols_to_load=cols_to_load,
            stratify_col=self.config.get("stratify_col")
        )

        # 2. Split
        # Using _stratify helper column from load_dataset if available
        stratify = df["_stratify"] if "_stratify" in df.columns else None
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=stratify, random_state=self.seed
        )

        # 3. Fit Processor
        processor = PGenProcessor(
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            multi_label_cols=list(MULTI_LABEL_COLS)
        )
        processor.fit(train_df)
        
        # 4. Create Datasets
        self.train_dataset = PGenDataset(
            processor.transform(train_df), 
            self.feature_cols, 
            self.target_cols, 
            MULTI_LABEL_COLS
        )
        self.val_dataset = PGenDataset(
            processor.transform(val_df), 
            self.feature_cols, 
            self.target_cols, 
            MULTI_LABEL_COLS
        )

        # Store encoder dims for model instantiation
        self.encoder_dims = {
            col: len(enc.classes_) for col, enc in processor.encoders.items()
        }
        logger.info(f"Data ready. Train: {len(train_df)}, Val: {len(val_df)}")

    # ==========================================================================
    # PARAMETER PARSING
    # ==========================================================================

    def _suggest_int(self, trial: optuna.Trial, name: str, args: List[Any]) -> int:
        # args: [low, high, step, log]
        low, high = args[0], args[1]
        step = args[2] if len(args) > 2 else 1
        log = args[3] if len(args) > 3 else False
        return trial.suggest_int(name, low, high, step=step, log=log)

    def _suggest_float(self, trial: optuna.Trial, name: str, args: Tuple[float, float]) -> float:
        # args: (low, high)
        low, high = args
        is_log = any(x in name for x in ["learning_rate", "weight_decay"])
        return trial.suggest_float(name, low, high, log=is_log)

    def _suggest_categorical(self, trial: optuna.Trial, name: str, choices: List[Any]) -> Any:
        return trial.suggest_categorical(name, choices)

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = self.config.get("params", {}).copy() # Start with defaults
        
        for name, space in self.optuna_space.items():
            try:
                if isinstance(space, list):
                    if not space: continue
                    
                    if space[0] == "int":
                        params[name] = self._suggest_int(trial, name, space[1:])
                    elif len(space) == 1:
                        params[name] = space[0] # Constant
                    else:
                        params[name] = self._suggest_categorical(trial, name, space)
                
                elif isinstance(space, tuple):
                    params[name] = self._suggest_float(trial, name, space)
                else:
                    params[name] = space
            
            except Exception as e:
                logger.error(f"Error suggesting param '{name}': {e}")
                raise
        return params

    # ==========================================================================
    # TRAINING LOOP (Objective)
    # ==========================================================================

    def objective(self, trial: optuna.Trial) -> Union[float, Tuple[float, float]]:
        # 1. Suggest Hyperparameters
        params = self._get_trial_params(trial)
        trial.set_user_attr("params", params)

        # 2. Efficient DataLoaders
        # Using pin_memory=True for faster GPU transfer if available
        train_loader = DataLoader(
            cast(PGenDataset, self.train_dataset), batch_size=params.get("batch_size", 64), 
            shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            cast(PGenDataset, self.val_dataset), batch_size=params.get("batch_size", 64), 
            shuffle=False, num_workers=0, pin_memory=True
        )

        # 3. Instantiate Model (Using Factory)
        curr_n_feats = {k: v for k, v in self.encoder_dims.items() if k in self.feature_cols}
        curr_n_targets = {k: v for k, v in self.encoder_dims.items() if k in self.target_cols}

        model = create_model(self.model_name, curr_n_feats, curr_n_targets, params).to(self.device)

        # 4. Setup Training Components
        
        # A) Uncertainty Loss (Optional)
        uncertainty_module = None
        if params.get("use_uncertainty_loss", False):
            if trial.number == 0: logger.info("Active: MultiTaskUncertaintyLoss")
            uncertainty_module = MultiTaskUncertaintyLoss(self.target_cols).to(self.device)

        # B) Optimizer & Scheduler
        trainable_params = list(model.parameters()) + (list(uncertainty_module.parameters()) if uncertainty_module else [])
        
        optimizer_name = params.get("optimizer_type", "adamw").lower()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(trainable_params, lr=params["learning_rate"], weight_decay=params["weight_decay"])
        else:
            optimizer = torch.optim.AdamW(trainable_params, lr=params["learning_rate"], weight_decay=params["weight_decay"])

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3)

        # 5. Train
        trainer = PGenTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            target_cols=self.target_cols,
            multi_label_cols=MULTI_LABEL_COLS,
            params=params,
            uncertainty_module=uncertainty_module
        )

        try:
            best_loss = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=self.epochs,
                patience=self.patience,
                trial=trial
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with params {params}: {e}")
            raise e

        # Multi-objective support placeholder (e.g., minimize loss, maximize accuracy)
        # Currently just returning loss as primary objective.
        return best_loss

    # ==========================================================================
    # EXECUTION & REPORTING
    # ==========================================================================

    def run(self):
        """Executes the full Optuna study."""
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")
        study_name = f"OPT_{self.model_name}_{timestamp}"
        storage_url = f"sqlite:///{DIRS['reports']}/optuna_reports/study_DBs/{study_name}.db"

        logger.info(f"Starting study: {study_name}")

        sampler = TPESampler(seed=self.seed, multivariate=True)
        
        if self.use_multi_objective:
            directions = ["minimize", "minimize"]
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                directions=directions,
                sampler=sampler,
                load_if_exists=True
            )
        else:
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="minimize",
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )

        # External Progress Bar
        with tqdm(total=self.n_trials, desc="Optuna Trials", colour="blue") as pbar:
            def progress_callback(study, trial):
                pbar.update(1)
                if study.best_trials:
                    best_val = study.best_trials[0].values[0]
                    pbar.set_postfix(best_loss=f"{best_val:.4f}")

            study.optimize(
                self.objective, 
                n_trials=self.n_trials, 
                callbacks=[progress_callback], 
                gc_after_trial=True
            )

        self._save_results(study, timestamp)
        return study

    def _save_results(self, study: optuna.Study, timestamp: str):
        """Generates JSON reports and plots."""
        logger.info("Generating reports...")
        
        # 1. Plots
        self._generate_plots(study, timestamp)

        # 2. JSON Report
        best_trials = study.best_trials
        base_name = f"report_{self.model_name}_{timestamp}"
        
        report_data = {
            "model": self.model_name,
            "best_trials": [
                {
                    "number": t.number,
                    "values": t.values,
                    "params": t.params,
                    "metrics": t.user_attrs
                } for t in best_trials
            ]
        }
        
        out_path = DIRS["reports"] / "optuna_reports" / f"{base_name}.json"
        with open(out_path, "w") as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Report saved to {out_path}")

    def _generate_plots(self, study, timestamp):
        """Safe wrapper for Optuna visualization."""
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )
            
            base_path = DIRS["reports"] / "figures" / f"{self.model_name}_{timestamp}"
            
            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(f"{base_path}_history.png")
            plt.close()

            if not self.use_multi_objective and len(study.trials) > 10:
                plt.figure(figsize=(10, 6))
                plot_param_importances(study)
                plt.tight_layout()
                plt.savefig(f"{base_path}_importance.png")
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_optuna_study(
    model_name: str, 
    csv_path: Union[str, Path],
    n_trials: int = 100
):
    tuner = OptunaTuner(model_name, csv_path, n_trials=n_trials)
    study = tuner.run()
    
    print("\n" + "="*50)
    print(f"Best Trial Params: {study.best_trials[0].params}")
    print("="*50 + "\n")
