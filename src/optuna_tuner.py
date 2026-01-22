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

"""
Optuna Hyperparameter Optimization with Multi-Objective Support.

This module implements a comprehensive hyperparameter optimization pipeline using Optuna
with support for multi-objective optimization. Key features:

- Multi-objective optimization (loss + F1 for critical tasks)
- Per-task metric tracking (F1, precision, recall)
- Proper handling of class imbalance with weighted loss
- Clinical priority weighting for task balancing
- Comprehensive reporting with JSON and visualization

Refactorizado para eficiencia de memoria:
1. Carga datos y crea Datasets una sola vez (evita Data Leakage y overhead IO).
2. Pasa referencias de los datasets a la función objetivo.
3. Soporta optimización mono y multi-objetivo.

References:
    - Optuna: https://optuna.readthedocs.io/
    - Multi-objective optimization: Deb et al., 2002
"""

import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.cfg.config import PROJECT_ROOT
from src.cfg.model_configs import MULTI_LABEL_COLUMN_NAMES, get_model_config

from src.data import PGenDataProcess, PGenDataset
from src.loss_functions import MultiTaskUncertaintyLoss
from src.model import DeepFM_PGenModel
from src.train import train_model
from src.utils.data import load_and_prep_dataset
from src.utils.training import create_optimizer, create_scheduler, create_task_criterions
# Configuración Global
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Constantes de Rutas
OPTUNA_OUTPUTS = Path(PROJECT_ROOT) / "reports" /"optuna_reports"
OPTUNA_FIGS = Path(PROJECT_ROOT) / "reports" / "figures"
OPTUNA_DBS = OPTUNA_OUTPUTS / "study_DBs"

# Asegurar existencia de directorios
for d in [OPTUNA_FIGS, OPTUNA_DBS]:
    d.mkdir(parents=True, exist_ok=True)


class OptunaTuner:
    """
    Clase orquestadora para la optimización de hiperparámetros.
    Encapsula datos, configuración y ciclo de vida de Optuna.
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

        # Cargar configuración del modelo
        self.config = get_model_config(model_name)
        self.feature_cols = [c.lower() for c in self.config["features"]]
        self.target_cols = [t.lower() for t in self.config["targets"]]
        
        # Validar que existe configuración de optimización
        self.optuna_space: Dict[str, List[Any]] = self.config.get("params_optuna", {})
        if not self.optuna_space:
            raise ValueError(f"No Optuna search space defined for model '{model_name}'")

        # Inicialización diferida de datos
        self.train_dataset: Optional[PGenDataset] = None
        self.val_dataset: Optional[PGenDataset] = None
        self.encoder_dims: Dict[str, int] = {}

        # Preparar datos al instanciar
        self._prepare_data()

    def _prepare_data(self):
        """Carga, procesa y cachea los datasets en memoria (One-time setup)."""
        logger.info(f"Preparando datos para {self.model_name}...")
        
        # Definir columnas necesarias
        cols_to_load = list(set(self.feature_cols + self.target_cols))
        
        # 1. Cargar y Limpiar Datos 
        df = load_and_prep_dataset(
            csv_path=self.csv_path,
            all_cols=cols_to_load,
            target_cols=self.target_cols,
            multi_label_targets=list(MULTI_LABEL_COLUMN_NAMES),
            stratify_cols=self.config.get("stratify_col") or self.config.get("stratify"),
        )

        # 2. Split
        train_df, val_df = train_test_split(
            df, test_size=0.2, stratify=df.get("stratify_col"), random_state=self.seed
        )

        # 3. Configurar Procesador (Clase)
        # AHORA: Se pasan las columnas en el __init__
        processor = PGenDataProcess(
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            multi_label_cols=list(MULTI_LABEL_COLUMN_NAMES)
        )

        # 4. Ajustar y Transformar
        processor.fit(train_df)
        
        # Crear Datasets
        self.train_dataset = PGenDataset(
            processor.transform(train_df), 
            self.feature_cols, 
            self.target_cols, 
            MULTI_LABEL_COLUMN_NAMES
        )
        self.val_dataset = PGenDataset(
            processor.transform(val_df), 
            self.feature_cols, 
            self.target_cols, 
            MULTI_LABEL_COLUMN_NAMES
        )

        # Guardar dimensiones para instanciar el modelo luego
        self.encoder_dims = {
            col: len(enc.classes_) for col, enc in processor.encoders.items()
        }
        logger.info(f"Datos listos. Train: {len(train_df)}, Val: {len(val_df)}")

    # ==========================================================================
    # PARSING DE PARÁMETROS (Clean Code: Separación de responsabilidades)
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
        # Heurística: si el nombre implica tasa o probabilidad pequeña, usar log
        is_log = any(x in name for x in ["learning_rate", "weight_decay"])
        
        return trial.suggest_float(name, low, high, log=is_log)

    def _suggest_categorical(self, trial: optuna.Trial, name: str, choices: List[Any]) -> Any:
        return trial.suggest_categorical(name, choices)

    def _get_trial_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params = {}
        for name, space in self.optuna_space.items():
            try:
                if isinstance(space, list):
                    # CASO 1: Lista vacía (error)
                    if not space:
                        continue

                    # CASO 2: Definición de Entero ["int", min, max, step]
                    if space[0] == "int":
                        params[name] = self._suggest_int(trial, name, space[1:])
                    
                    # CASO 3: Valor Fijo (Lista de 1 elemento)
                    # Optuna explota si le pides sugerir de una lista de 1 solo valor repetidamente
                    elif len(space) == 1:
                        params[name] = space[0]
                        # Opcional: Guardarlo como atributo fijo para que conste en el log
                        trial.set_user_attr(f"fixed_{name}", space[0])

                    # CASO 4: Categórico Real (Más de 1 opción)
                    else:
                        params[name] = self._suggest_categorical(trial, name, space)
                
                elif isinstance(space, tuple):
                    params[name] = self._suggest_float(trial, name, space)
                
                else:
                    # Fallback constante (si en el TOML pusiste un valor directo sin lista)
                    params[name] = space
            
            except Exception as e:
                logger.error(f"Error sugiriendo parámetro '{name}': {e}")
                raise
        return params

    # ==========================================================================
    # LOOP DE ENTRENAMIENTO (Objective)
    # ==========================================================================

    def objective(self, trial: optuna.Trial) -> Union[float, Tuple[float, float]]:
        # 1. Sugerir Hiperparámetros
        params = self._get_trial_params(trial)
        trial.set_user_attr("params", params)

        # 2. DataLoaders Eficientes
        train_loader = DataLoader(
            cast(PGenDataset, self.train_dataset), batch_size=params.get("batch_size", 64), 
            shuffle=True, num_workers=0, pin_memory=True
        )
        val_loader = DataLoader(
            cast(PGenDataset, self.val_dataset), batch_size=params.get("batch_size", 64), 
            shuffle=False, num_workers=0, pin_memory=True
        )

        # 3. Instanciar Modelo
        curr_n_feats = {k: v for k, v in self.encoder_dims.items() if k in self.feature_cols}
        curr_n_targets = {k: v for k, v in self.encoder_dims.items() if k in self.target_cols}

        model = DeepFM_PGenModel(
            n_features=curr_n_feats,
            target_dims=curr_n_targets,
            embedding_dim=params["embedding_dim"],
            hidden_dim=params["hidden_dim"],
            dropout_rate=params["dropout_rate"],
            n_layers=params["n_layers"],
            activation_function=params.get("activation_function", "gelu"),
            fm_hidden_dim=params.get("fm_hidden_dim", 64),
            # ... 
            attention_dim_feedforward=params.get("attention_dim_feedforward"),
            attention_dropout=params.get("attention_dropout", 0.1),
            num_attention_layers=params.get("num_attention_layers", 1),
            use_batch_norm=params.get("use_batch_norm", False),
            use_layer_norm=params.get("use_layer_norm", False),
            embedding_dropout=params.get("embedding_dropout", 0.0),
        ).to(self.device)

        # 4. Configurar Training Components
        # a) Loss Functions (diccionario de losses individuales)
        loss_fns_dict = create_task_criterions(
            self.target_cols, MULTI_LABEL_COLUMN_NAMES, params, self.device
        )
        
        # b) Uncertainty Loss Wrapper (Opcional según params)
        uncertainty_module = None
        if params.get("use_uncertainty_loss", False):
            logger.info("Activando MultiTaskUncertaintyLoss (Kendall & Gal)") if trial.number == 0 else None
            uncertainty_module = MultiTaskUncertaintyLoss(self.target_cols).to(self.device)

        # c) Optimizer y Scheduler
        optimizer = create_optimizer(model, params, uncertainty_module)
        scheduler = create_scheduler(optimizer, params)

        # e) Empaquetar criterions como lista para compatibilidad con train_model antiguo

        criterions_list: List[Any] = [loss_fns_dict[t] for t in self.target_cols]
        criterions_list.append(optimizer)

        # 5. Entrenar
        try:
            best_loss, best_accs = train_model(
                train_loader=train_loader,
                val_loader=val_loader,
                model=model,
                criterions=criterions_list,
                epochs=self.epochs,
                patience=self.patience,
                model_name=self.model_name,
                feature_cols=self.feature_cols,
                target_cols=self.target_cols,
                device=self.device,
                scheduler=scheduler,
                multi_label_cols=MULTI_LABEL_COLUMN_NAMES,
                trial=trial,
                return_per_task_losses=False,
                progress_bar=False,
                uncertainty_loss_module=uncertainty_module 
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} falló con params {params}: {e}")
            raise e

        # Reportar métricas extra
        avg_acc = sum(best_accs) / len(best_accs) if best_accs else 0.0
        trial.set_user_attr("avg_accuracy", avg_acc)

        if self.use_multi_objective:
            return best_loss, -avg_acc
        
        return best_loss

    # ==========================================================================
    # EJECUCIÓN Y REPORTING
    # ==========================================================================

    def run(self):
        """Ejecuta el estudio de Optuna completo."""
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M")
        study_name = f"OPT_{self.model_name}_{timestamp}"
        storage_url = f"sqlite:///{OPTUNA_DBS}/{study_name}.db"

        logger.info(f"Iniciando estudio: {study_name}")

        # Configurar Sampler y Pruner
        sampler = TPESampler(seed=self.seed, multivariate=True)
        
        if self.use_multi_objective:
            directions = ["minimize", "minimize"]
            pruner = None
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                directions=directions,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
        else:
            direction = "minimize"
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )

        # Barra de progreso externa
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

        # Guardar resultados
        self._save_results(study, timestamp)
        return study

    def _save_results(self, study: optuna.Study, timestamp: str):
        """Genera gráficos y reportes JSON/TXT."""
        logger.info("Generando reportes...")
        
        # 1. Gráficos
        self._generate_plots(study, timestamp)

        # 2. Texto y JSON
        best_trials = self._get_pareto_front(study)
        base_name = f"report_{self.model_name}_{timestamp}"
        
        # JSON Data
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
        
        with open(OPTUNA_OUTPUTS / f"{base_name}.json", "w") as f:
            json.dump(report_data, f, indent=2)
            
        logger.info(f"Reportes guardados en {OPTUNA_OUTPUTS}")

    def _get_pareto_front(self, study: optuna.Study, n: int = 5):
        valid = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid: 
            return []
        # En multi-obj devuelve frente pareto, en single devuelve best trial
        return study.best_trials[:n]

    def _generate_plots(self, study, timestamp):
        """Wrapper seguro para Matplotlib."""
        try:
            from optuna.visualization.matplotlib import (
                plot_optimization_history,
                plot_param_importances,
            )
        except Exception as e:
            logger.warning(f"No se pudieron importar utilidades de optuna para gráficos: {e}")
            return

        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning(f"No se pudo importar matplotlib: {e}")
            return

        try:
            base_path = OPTUNA_FIGS / f"{self.model_name}_{timestamp}"
            
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
            logger.warning(f"No se pudieron generar gráficos: {e}")

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
    print(f"Mejor Trial Params: {study.best_trials[0].params}")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Ejemplo
    # run_optuna_study("DeepFM_V1", "data/processed.tsv")
    pass