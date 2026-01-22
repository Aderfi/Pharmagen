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

import logging
import os #noqa
import sys
from pathlib import Path
from typing import Any

import tomli

#from .config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).parent
GLOBAL_CONFIG_FILE = CONFIG_DIR / "config.toml"
MODELS_FILE = CONFIG_DIR / "models.toml"
# =============================================================================
# UTILIDAD DE CONVERSIÓN (TOML -> OPTUNA TYPES)
# =============================================================================
def _parse_optuna_value(val: Any) -> Any:
    """
    Convierte las listas de TOML en los tipos que OptunaTuner espera.
    - [0.1, 0.5] (2 floats/ints) -> tuple (0.1, 0.5) -> Para suggest_float
    - ["int", 1, 10] -> list (se queda igual) -> Para suggest_int
    - [128, 256, 512] (>2 elementos) -> list (se queda igual) -> Para suggest_categorical
    """
    if isinstance(val, list):
        # Caso 1: Definición explícita de entero ["int", min, max...]
        if len(val) > 0 and isinstance(val[0], str) and val[0] == "int":
            return val
        
        # Caso 2: Rango Flotante Implícito [min, max]
        # Heurística: Si tiene exactamente 2 elementos y ambos son números
        if len(val) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in val):
            return tuple(val)  # <--- LA CLAVE: Convertir a tupla
            
        # Caso 3: Categorical (Cualquier otra lista)
        return val
        
    return val

def _parse_optuna_section(optuna_dict: dict[str, Any]) -> dict[str, Any]:
    """Aplica la conversión a todo el diccionario de optuna."""
    return {k: _parse_optuna_value(v) for k, v in optuna_dict.items()}

# =============================================================================
# LECTURA Y FUSIÓN DE CONFIGURACIÓN
# =============================================================================
def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if sys.version_info >= (3, 11):
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    else:
        with open(path, "rb") as f:
            return tomli.load(f)

def get_model_names() -> list[str]:
    """Devuelve lista de modelos disponibles en models.toml."""
    models_data = _load_toml(MODELS_FILE)
    return list(models_data.keys())

def get_model_config(model_name: str) -> dict[str, Any]:
    """
    Genera la configuración FINAL del modelo aplicando herencia y parseo de tipos.
    """
    # 1. Cargar Configs
    global_conf = _load_toml(GLOBAL_CONFIG_FILE)
    models_data = _load_toml(MODELS_FILE)
    
    if model_name not in models_data:
        raise ValueError(f"Model '{model_name}' not defined in models.toml")
    
    defaults = global_conf.get("params", {})
    project_conf = global_conf.get("project", {})
    specific_model_data = models_data[model_name]
    
    # 2. Fusión (Merge)
    final_config = defaults.copy()
    final_config.update(project_conf)
    
    # Extraer subsecciones
    optuna_params_raw = specific_model_data.pop("optuna", {})
    weights = specific_model_data.pop("weights", None)
    specific_params = specific_model_data.pop("params", {})
    
    # Actualizar configs
    final_config.update(specific_model_data) # Claves raíz (features, targets)
    final_config.update(specific_params)     # Sobreescrituras de params
    
    # 3. PROCESAMIENTO CRÍTICO DE OPTUNA
    # Convertimos las listas [min, max] de TOML a tuplas (min, max) de Python
    final_config["params_optuna"] = _parse_optuna_section(optuna_params_raw)
    
    if weights:
        final_config["manual_task_weights_dict"] = weights

    # Validación básica
    if "features" not in final_config or "targets" not in final_config:
        raise ValueError(f"Model '{model_name}' missing 'features' or 'targets'.")

    return final_config

# =============================================================================
# CONSTANTES EXPORTADAS (Helpers)
# =============================================================================

# Cargamos configuración global para exportar constantes comunes si se necesitan
_global = _load_toml(GLOBAL_CONFIG_FILE)
MULTI_LABEL_COLUMN_NAMES = set(_global.get("project", {}).get("multi_label_cols", []))
MODEL_REGISTRY = _load_toml(MODELS_FILE) # Exportamos el raw por si acaso

# Registro de modelos (Lazy loading para no leer disco al importar si no se usa)
# Se accede vía get_model_config(name)

# =============================================================================
# CLI SELECTION
# =============================================================================

def select_model(prompt: str = "Selecciona el modelo a entrenar:", default_choice: int | None = None) -> str:
    """CLI interactiva para elegir modelo."""
    options = get_model_names()
    if default_choice is not None and 1 <= default_choice <= len(options):
        selected = options[default_choice - 1]
        print(f"Selected by default: {selected}")
        return selected

    if not options:
        print("Error: No hay modelos definidos en models.toml")
        sys.exit(1)

    print("\n" + "═"*50)
    print(f" MODELOS DISPONIBLES ({len(options)})")
    print("═"*50)
    for i, name in enumerate(options, 1):
        print(f"  {i}. {name}")
    print("═"*50)

    while True:
        choice = input(f"\n{prompt} (1-{len(options)}): ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                selected = options[idx]
                print(f"Selected: {selected}")
                return selected
        print("❌ Selección inválida.")

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).parent.parent.parent
    model = select_model(default_choice=2)
    cfg = get_model_config(model)

    for i in cfg.keys():
        print(f"\n{i}: {cfg[i]}")