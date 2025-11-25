# Pharmagen - Configuration Manager
# Adheres to Zen of Python: Explicit is better than implicit.

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# TOML Parser (Compatibility wrapper)
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

# =============================================================================
# 1. CONSTANTS & PATHS
# =============================================================================

# Root is 3 levels up from this file (src/cfg/manager.py -> src/cfg -> src -> root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(__file__).parent

# Load Core Config Files
try:
    with open(CONFIG_DIR / "paths.toml", "rb") as f:
        _PATHS_CFG = tomllib.load(f)
    with open(CONFIG_DIR / "config.toml", "rb") as f:
        _GLOBAL_CFG = tomllib.load(f)
    with open(CONFIG_DIR / "models.toml", "rb") as f:
        _MODELS_CFG = tomllib.load(f)
except FileNotFoundError as e:
    sys.exit(f"CRITICAL: Missing configuration file: {e}")

def _resolve(path_str: str) -> Path:
    """Resolves paths relative to PROJECT_ROOT."""
    return PROJECT_ROOT / path_str

# Exported Constants (Flat structure for easy import)
METADATA = _GLOBAL_CFG.get("metadata", {})
PROJECT_NAME = METADATA.get("project_name", "Pharmagen")
VERSION = METADATA.get("version", "0.0.0")
DATE_STAMP = datetime.now().strftime("%Y_%m_%d")

# Directory Map
DIRS = {
    "base": PROJECT_ROOT, # Added explicit base
    "data": _resolve(_PATHS_CFG["base"]["data"]),
    "logs": _resolve(_PATHS_CFG["base"]["logs"]),
    "results": _resolve(_PATHS_CFG["base"]["results"]),
    "reports": _resolve(_PATHS_CFG["base"]["reports"]),
    "models": _resolve(_PATHS_CFG["models"]["models_saved"]),
    "encoders": _resolve(_PATHS_CFG["models"]["encoders"]),
}

# Ensure directories exist
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Helpers
MULTI_LABEL_COLS = set(_GLOBAL_CFG.get("project", {}).get("multi_label_cols", []))

# =============================================================================
# 2. MODEL CONFIGURATION LOGIC
# =============================================================================

def get_available_models() -> List[str]:
    return list(_MODELS_CFG.keys())

def _parse_optuna_param(val: Any) -> Any:
    """
    Parses TOML lists into Python tuples/types for Optuna.
    [min, max] -> (min, max)
    ["int", min, max] -> ["int", min, max] (kept as list for specific handling)
    """
    if isinstance(val, list):
        # Check for explicit type definition or range
        if len(val) > 0 and val[0] == "int":
            return val
        if len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
            return tuple(val)
    return val

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Returns a merged configuration dictionary for a specific model.
    Priority: Model Config > Global Defaults.
    """
    if model_name not in _MODELS_CFG:
        raise ValueError(f"Model '{model_name}' not found in models.toml")

    # 1. Start with defaults
    config = _GLOBAL_CFG.get("params", {}).copy()
    config.update(_GLOBAL_CFG.get("project", {}))

    # 2. Update with specific model config
    model_data = _MODELS_CFG[model_name].copy()
    
    # Flatten nested specific params
    if "params" in model_data:
        config.update(model_data.pop("params"))
    
    # Process Optuna params if present
    if "optuna" in model_data:
        config["params_optuna"] = {
            k: _parse_optuna_param(v) for k, v in model_data.pop("optuna").items()
        }
    
    config.update(model_data)
    
    # 3. Validation
    required_keys = ["features", "targets"]
    if not all(k in config for k in required_keys):
        raise ValueError(f"Model config requires {required_keys}")

    return config

if __name__ == "__main__":
    print(f"Pharmagen Config Manager v{VERSION}")
    print(f"Root: {PROJECT_ROOT}")
    print(f"Available Models: {get_available_models()}")