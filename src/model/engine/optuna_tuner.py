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
"""
optuna_tuner.py

Manages Hyperparameter Optimization (HPO) using Optuna.
Implements a clean 'Orchestrator' pattern to separate the search logic
from the model training loop.
"""

from dataclasses import dataclass
import gc
import json
import logging
from typing import Any

import matplotlib.pyplot as plt
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from torch.utils.data import DataLoader

from src.cfg.manager import DIRS
from src.data.data_handler import (
    DataConfig,
    PGenDataset,
    PGenProcessor,
    load_and_clean_dataset,
)
from src.interface.ui import ConsoleIO
from src.model.architecture.deep_fm import ModelConfig, PharmagenDeepFM
from src.model.engine.trainer import PGenTrainer, TrainerConfig

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.INFO)

# =============================================================================
# 1. OPTUNA CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TunerConfig:
    """
    Configuration specific to the Optimization Orchestration.
    """

    study_name: str
    n_trials: int = 50
    storage_url: str = "sqlite:///optuna_study.db"
    direction: str = "minimize"
    seed: int = 42

    # Pruning logic
    use_pruning: bool = True
    n_startup_trials: int = 10
    n_warmup_steps: int = 5


# =============================================================================
# 2. THE OPTIMIZATION ORCHESTRATOR
# =============================================================================


class OptunaOrchestrator:
    """
    Bridges the gap between Optuna's Search Space and the System's Config Objects.
    Manages the lifecycle of Data -> Config -> Training -> Cleanup.

    Attributes:
        tuner_cfg (TunerConfig): Configuration for the study (storage, trials).
        base_data_cfg (DataConfig): Configuration for data loading.
        search_space (dict): Definition of hyperparameter ranges.
    """

    def __init__(
        self,
        tuner_cfg: TunerConfig,
        data_cfg: DataConfig,
        search_space: dict[str, list[Any]],
        device: str = "cuda",
    ):
        self.tuner_cfg = tuner_cfg
        self.base_data_cfg = data_cfg
        self.search_space = search_space
        self.device = device

        # Pre-loaded Data Artifacts (Shared across trials)
        self.train_dataset: PGenDataset | None = None
        self.val_dataset: PGenDataset | None = None
        self.feature_dims: dict[str, int] = {}
        self.target_dims: dict[str, int] = {}

        # Initialize Data
        self._warmup_data()

    def _warmup_data(self):
        """
        Loads and processes data ONCE.
        Reuse the same PGenDataset tensors for all trials to save RAM/Time.
        """
        logger.info("--- Warming up Data Engine ---")

        # 1. Load & Clean
        df = load_and_clean_dataset(self.base_data_cfg)

        # 2. Deterministic split
        val_mask = df.sample(frac=0.2, random_state=self.tuner_cfg.seed).index
        train_df = df.drop(val_mask)
        val_df = df.loc[val_mask]

        # 3. Fit Processor
        processor = PGenProcessor(
            config={
                "features": self.base_data_cfg.feature_cols,
                "targets": self.base_data_cfg.target_cols,
            },
            multi_label_cols=self.base_data_cfg.multi_label_cols,
        )
        processor.fit(train_df)

        # 4. Transform to Tensors
        self.train_dataset = PGenDataset(
            processor.transform(train_df),
            self.base_data_cfg.feature_cols,
            self.base_data_cfg.target_cols,
            set(self.base_data_cfg.multi_label_cols),
        )
        self.val_dataset = PGenDataset(
            processor.transform(val_df),
            self.base_data_cfg.feature_cols,
            self.base_data_cfg.target_cols,
            set(self.base_data_cfg.multi_label_cols),
        )

        # 5. Extract Dimensions for Model Config
        all_dims = {col: len(enc.classes_) for col, enc in processor.encoders.items()}
        self.feature_dims = {
            k: v for k, v in all_dims.items() if k in self.base_data_cfg.feature_cols
        }
        self.target_dims = {
            k: v for k, v in all_dims.items() if k in self.base_data_cfg.target_cols
        }

        logger.info("Data Ready. Train Samples: %d | Val Samples: %d", len(train_df), len(val_df))

    # ==========================================================================
    # DYNAMIC CONFIGURATION BUILDER
    # ==========================================================================

    def _suggest_parameter(self, trial: optuna.Trial, name: str, spec: list[Any]) -> Any:
        """
        Parses the search space list syntax:
        ['int', low, high, step]
        ['float', low, high, log]
        ['cat', choice1, choice2...]
        ['const', value]
        """
        param_type = spec[0]

        if param_type == "int":
            return trial.suggest_int(name, spec[1], spec[2], step=spec[3] if len(spec) > 3 else 1)

        if param_type == "float":
            log_scale = spec[3] if len(spec) > 3 else False
            return trial.suggest_float(name, spec[1], spec[2], log=log_scale)

        if param_type == "cat":
            return trial.suggest_categorical(name, spec[1:])

        if param_type == "const":
            return spec[1]

        # Fallback for direct categorical list shorthand
        return trial.suggest_categorical(name, spec)

    def _build_configs(self, trial: optuna.Trial) -> tuple[ModelConfig, TrainerConfig, int]:
        """Constructs strict configuration objects from Trial suggestions."""
        params: dict[str, Any] = {}
        for key, spec in self.search_space.items():
            params[key] = self._suggest_parameter(trial, key, spec)

        # Mapeo norm_type → flags individuales de ModelConfig
        # El TOML usa norm_type="bn"/"ln"/"none" para evitar la combinación
        # BN+LN simultánea que sería semánticamente incorrecta.
        norm_type = params.get("norm_type", "bn")
        use_bn = norm_type == "bn"
        use_ln = norm_type == "ln"

        model_config = ModelConfig(
            n_features=self.feature_dims,
            target_dims=self.target_dims,
            embedding_dim=params.get("embedding_dim", 64),
            embedding_dropout=params.get("embedding_dropout", 0.1),
            hidden_dim=params.get("hidden_dim", 256),
            n_layers=params.get("n_layers", 3),
            dropout_rate=params.get("dropout_rate", 0.2),
            activation=params.get("activation", "gelu"),
            use_batch_norm=use_bn,
            use_layer_norm=use_ln,
            use_transformer=params.get("use_transformer", True),
            attn_dim_feedforward=params.get("attn_dim_feedforward", 512),
            attn_heads=params.get("attn_heads", 4),
            num_attn_layers=params.get("num_attn_layers", 2),
            fm_hidden_dim=params.get("fm_hidden_dim", 64),
            fm_hidden_layers=params.get("fm_hidden_layers", 1),
            fm_dropout=params.get("fm_dropout", 0.1),
        )

        trainer_config = TrainerConfig(
            n_epochs=params.get("epochs", 50),
            patience=params.get("patience", 10),
            learning_rate=params.get("learning_rate", 1e-3),
            weight_decay=params.get("weight_decay", 1e-4),
            grad_clip_norm=params.get("grad_clip_norm", 1.0),
            device=self.device,
            use_amp=True,
            ml_loss_type=params.get("ml_loss_type", "asymmetric"),
            mc_loss_type=params.get("mc_loss_type", "focal"),
            label_smoothing=params.get("label_smoothing", 0.0),
            focal_gamma=params.get("focal_gamma", 2.0),
            asl_gamma_neg=params.get("asl_gamma_neg", 4.0),
            asl_gamma_pos=params.get("asl_gamma_pos", 1.0),
        )

        batch_size = params.get("batch_size", 128)
        return model_config, trainer_config, batch_size

    # ==========================================================================
    # OBJECTIVE FUNCTION
    # ==========================================================================

    def objective(self, trial: optuna.Trial) -> float:
        """Core Optimization Loop for a single trial."""
        gc.collect()
        torch.cuda.empty_cache()

        # 1. Build Configuration
        model_cfg, trainer_cfg, batch_size = self._build_configs(trial)

        # 2. Init DataLoaders
        assert self.train_dataset is not None, "train_dataset was not initialized"
        assert self.val_dataset is not None, "val_dataset was not initialized"

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # 3. Init Model & Optimization
        model = PharmagenDeepFM(model_cfg)

        uncertainty_module = None
        trainable_params = list(model.parameters())
        if uncertainty_module is not None:
            trainable_params += list(uncertainty_module.parameters())

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=trainer_cfg.learning_rate,
            weight_decay=trainer_cfg.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )

        # 4. Init Trainer
        trainer = PGenTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=trainer_cfg,
            target_cols=self.base_data_cfg.target_cols,
            multi_label_cols=set(self.base_data_cfg.multi_label_cols),
            uncertainty_module=uncertainty_module,
        )

        # 5. Execute Training
        try:
            best_loss = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                trial=trial,
            )
            return best_loss

        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.exception("Trial %d failed: %s", trial.number, e)
            return float("inf")

    # ==========================================================================
    # RUNNER
    # ==========================================================================

    def run(self) -> optuna.Study:
        """Starts the optimization study."""
        (DIRS["reports"] / "optuna_db").mkdir(parents=True, exist_ok=True)
        (DIRS["reports"] / "figures").mkdir(parents=True, exist_ok=True)

        logger.info("--- Starting Optuna Study: %s ---", self.tuner_cfg.study_name)

        pruner = (
            MedianPruner(
                n_startup_trials=self.tuner_cfg.n_startup_trials,
                n_warmup_steps=self.tuner_cfg.n_warmup_steps,
            )
            if self.tuner_cfg.use_pruning
            else None
        )

        study = optuna.create_study(
            study_name=self.tuner_cfg.study_name,
            storage=self.tuner_cfg.storage_url,
            direction=self.tuner_cfg.direction,
            sampler=TPESampler(seed=self.tuner_cfg.seed),
            pruner=pruner,
            load_if_exists=True,
        )

        study.optimize(
            self.objective,
            n_trials=self.tuner_cfg.n_trials,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        self._export_results(study)
        return study

    def _export_results(self, study: optuna.Study) -> None:
        best_trial = study.best_trial
        clean_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        best_five: list[optuna.trial.FrozenTrial] = sorted(clean_trials, key=lambda t: t.value)[:5]



        ConsoleIO.print_divider("=", 40)
        ConsoleIO.print_info(f"Best Value: {best_trial.value:.4f}")
        ConsoleIO.print_info("Best Params:")
        for key, value in best_trial.params.items():
            ConsoleIO.print_info(f"  {key}: {value}")
        ConsoleIO.print_divider("=", 40)

        report_path = DIRS["reports"] / f"{self.tuner_cfg.study_name}_best.json"

        js_report = {f"Trial {best_five[i].number}": best_five[i].params for i in range(min(5, len(best_five)))}

        with open(report_path, "w") as f:
            json.dump(js_report, f, indent=4)

        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )

            fig_path = DIRS["reports"] / "figures"
            plt.switch_backend("Agg")

            plt.figure(figsize=(10, 6))
            plot_optimization_history(study)
            plt.savefig(fig_path / f"{self.tuner_cfg.study_name}_history.png")
            plt.close()

            plt.figure(figsize=(10, 6))
            plot_param_importances(study)
            plt.savefig(fig_path / f"{self.tuner_cfg.study_name}_importance.png")
            plt.close()

        except ImportError:
            ConsoleIO.print_warning("Matplotlib not installed or failed. Skipping plots.")

def sito_export(study: optuna.Study) -> None:
        best_trial = study.best_trial

        clean_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
        best_five: list[optuna.trial.FrozenTrial] = sorted(clean_trials, key=lambda t: t.value)[:5]


        ConsoleIO.print_divider("=", 40)
        ConsoleIO.print_info(f"Best Value: {best_trial.value:.4f}")
        ConsoleIO.print_info("Best Params:")
        for key, value in best_trial.params.items():
            ConsoleIO.print_info(f"  {key}: {value}")
        ConsoleIO.print_divider("=", 40)

        js_report = {f"Trial {best_five[i].number}": best_five[i].params for i in range(min(5, len(best_five)))}

        report_path = DIRS["reports"] / f"{study.study_name}_best.json"
        with open(report_path, "w") as f:
            json.dump(js_report, f, indent=4)

if __name__ == "__main__":
    estudio = optuna.study.load_study(storage="sqlite:///reports/optuna_study.db", study_name="Pheno-Dir-Effect_study")

    sito_export(estudio)
