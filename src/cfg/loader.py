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

# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani

import logging
from typing import Any, Dict, List, Optional
from pathlib import Path
import tomli

from src.cfg.manager import CFG_FILE, MODELS_FILE

logger = logging.getLogger(__name__)

class ModelConfigLoader:
    """
    Responsible for loading, merging, and parsing model configurations.
    """
    
    @staticmethod
    def _load_toml(path: Path) -> Dict[str, Any]:
        with open(path, "rb") as f:
            return tomli.load(f)

    @staticmethod
    def get_available_models() -> List[str]:
        data = ModelConfigLoader._load_toml(MODELS_FILE)
        return list(data.keys())

    @classmethod
    def load_config(cls, model_name: str) -> Dict[str, Any]:
        """
        Loads and merges configuration: Defaults -> Project -> Model Specific.
        """
        global_conf = cls._load_toml(CFG_FILE)
        models_data = cls._load_toml(MODELS_FILE)

        if model_name not in models_data:
            raise ValueError(f"Model '{model_name}' not found in {MODELS_FILE}")

        # 1. Base Layer
        final_config = global_conf.get("defaults", {}).get("params", {}).copy()
        final_config.update(global_conf.get("project", {}))

        # 2. Specific Layer
        model_specific = models_data[model_name]
        
        # Separate special sections
        optuna_params = model_specific.pop("optuna", {})
        params_override = model_specific.pop("params", {})
        
        # Merge Top-level (features, targets)
        final_config.update(model_specific)
        # Merge Params
        final_config.update(params_override)
        
        # 3. Parse Optuna
        final_config["params_optuna"] = cls._parse_optuna_section(optuna_params)

        cls._validate(final_config, model_name)
        return final_config

    @staticmethod
    def _parse_optuna_section(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
        parsed = {}
        for k, v in raw_dict.items():
            # Convert list [0.1, 0.5] to tuple for ranges
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] != "int":
                parsed[k] = tuple(v)
            else:
                parsed[k] = v
        return parsed

    @staticmethod
    def _validate(config: Dict[str, Any], name: str):
        required = ["features", "targets"]
        for r in required:
            if r not in config:
                raise ValueError(f"Config for '{name}' is missing required field: '{r}'")