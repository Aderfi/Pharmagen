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

import logging
from pathlib import Path
import sys
import tomllib
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_toml(filename: str) -> dict[str, Any]:
    """Safe TOML loader helper."""
    path = CONFIG_DIR / filename
    if not path.exists():
        logger.critical("Configuration file missing: %s", path)
        sys.exit(1)
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception as e:
        logger.critical("Failed to parse %s: %s", filename, e)
        sys.exit(1)


def _resolve_paths(raw_paths: dict[str, Any]) -> dict[str, Path]:
    """Recursively resolves string paths in TOML to absolute Path objects."""
    resolved = {}
    base_section = raw_paths.get("base", {})  # noqa

    for _section, paths in raw_paths.items():
        if not isinstance(paths, dict):
            continue
        for key, relative_path in paths.items():
            resolved[key] = PROJECT_ROOT / str(relative_path)

    resolved["base"] = PROJECT_ROOT
    return resolved


# Initialization
_PATHS_RAW = _load_toml("paths.toml")
_GLOBAL_RAW = _load_toml("config.toml")
_MODELS_RAW = _load_toml("models.toml")

METADATA = _GLOBAL_RAW.get("metadata", {})
PROJECT_NAME = METADATA.get("project_name", "Pharmagen")
VERSION = METADATA.get("version", "0.0.0")

DIRS = _resolve_paths(_PATHS_RAW)
for d in DIRS.values():
    if d.suffix == "":
        d.mkdir(parents=True, exist_ok=True)

MULTI_LABEL_COLS = set(_GLOBAL_RAW.get("project", {}).get("multi_label_cols", []))


def get_available_models() -> list[str]:
    """
    Returns list of defined model names available for training.

    Returns:
        list[str]: Keys found in models.toml.

    Example:
        >>> get_available_models()
        ['Phenotype_Effect_Outcome', 'Features-Phenotype']
    """
    return list(_MODELS_RAW.keys())


def get_model_config(model_name: str) -> dict[str, Any]:
    """
    Returns the fully merged configuration for a requested model.

    Merges configuration layers in order:
    1. Global Defaults (config.toml)
    2. Project Settings (config.toml)
    3. Model Specifics (models.toml)

    Args:
        model_name (str): The exact name key in models.toml.

    Returns:
        dict: A dictionary with keys 'architecture', 'training', 'data', etc.

    Raises:
        ValueError: If model_name is not found or config is invalid.

    Example:
        >>> cfg = get_model_config("Phenotype_Effect_Outcome")
        >>> cfg['training']['batch_size']
        64
    """
    if model_name not in _MODELS_RAW:
        raise ValueError(f"Model '{model_name}' not defined in models.toml")

    # 1. Global Defaults
    config = _GLOBAL_RAW.get("defaults", {}).copy()

    # 2. Project Settings
    config["project"] = _GLOBAL_RAW.get("project", {})

    # 3. Model Specifics (Deep Merge)
    model_spec = _MODELS_RAW[model_name].copy()

    for section in ["data", "architecture", "training", "loss", "optuna"]:
        if section in model_spec:
            if section in config:
                config[section].update(model_spec[section])
            else:
                config[section] = model_spec[section]

    _validate_config(config, model_name)
    return config


def _parse_optuna_section(raw: dict[str, Any]) -> dict[str, Any]:
    """Parses Optuna ranges from TOML lists to Python tuples."""
    parsed = {}
    for k, v in raw.items():
        if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            parsed[k] = tuple(v)
        else:
            parsed[k] = v
    return parsed


def _validate_config(cfg: dict[str, Any], name: str):
    """Ensures critical keys exist in the configuration."""
    if "data" not in cfg:
        raise ValueError(f"Model '{name}' missing [data] section.")

    missing = [k for k in ["features", "targets"] if k not in cfg["data"]]
    if missing:
        raise ValueError(f"Model '{name}' [data] section missing: {missing}")


if __name__ == "__main__":
    print(f"✅ Loaded Manager for {PROJECT_NAME} v{VERSION}")
