#!/usr/bin/env python3
"""
Pharmagen - Environment Initialization Script
Unifies environment creation (venv/conda), dependency installation, and project scaffolding.
"""

import sys
import shutil
import subprocess
import argparse
from pathlib import Path

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.10"
ENV_NAME = "pharmagen_env"

def check_python_version():
    """Verifies current python version matches target."""
    ver = sys.version_info
    target = tuple(map(int, PYTHON_VERSION.split(".")))
    if ver.major != target[0] or ver.minor != target[1]:
        print(f"⚠️  Warning: Running with Python {ver.major}.{ver.minor}. Target is {PYTHON_VERSION}.")
    else:
        print(f"✅ Python version compatible: {ver.major}.{ver.minor}")

def create_scaffolding():
    """Creates necessary directory structure."""
    print("📂 Creating directory structure...")
    dirs = [
        PROJECT_ROOT / "data/raw",
        PROJECT_ROOT / "data/processed",
        PROJECT_ROOT / "data/dicts",
        PROJECT_ROOT / "data/ref_genome",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "cache",
        PROJECT_ROOT / "reports/figures",
        PROJECT_ROOT / "reports/optuna_reports",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "src/pgen_model/models",
        PROJECT_ROOT / "src/pgen_model/encoders",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print("✅ Directories ready.")

def setup_conda():
    """Creates and configures a Conda environment."""
    if not shutil.which("conda"):
        print("❌ Error: 'conda' not found in PATH.")
        return False
    
    print(f"🐍 Creating Conda env: {ENV_NAME}...")
    try:
        subprocess.run(
            ["conda", "create", "-n", ENV_NAME, f"python={PYTHON_VERSION}", "-y"], 
            check=True
        )
        print("📦 Installing project in editable mode...")
        # Use conda run to install pip deps inside the env
        subprocess.run(
            ["conda", "run", "-n", ENV_NAME, "pip", "install", "-e", ".[train]"],
            check=True, cwd=PROJECT_ROOT
        )
        print(f"✅ Setup Complete! Activate with: conda activate {ENV_NAME}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Conda setup failed: {e}")
        return False

def setup_venv():
    """Creates and configures a standard Venv."""
    venv_dir = PROJECT_ROOT / "venv"
    if venv_dir.exists():
        print(f"⚠️  Directory {venv_dir} exists. Skipping creation.")
    else:
        print(f"🐍 Creating Venv at {venv_dir}...")
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    
    # Determine pip path
    if sys.platform == "win32":
        pip_exe = venv_dir / "Scripts" / "pip.exe"
    else:
        pip_exe = venv_dir / "bin" / "pip"
        
    print("📦 Installing project in editable mode...")
    try:
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_exe), "install", "-e", ".[train]", "-e", ".[dev]"], check=True, cwd=PROJECT_ROOT)
        
        activate_cmd = ".\venv\Scripts\activate" if sys.platform == "win32" else "source ./venv/bin/activate"
        print(f"✅ Setup Complete! Activate with: {activate_cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Venv setup failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Pharmagen Environment Setup")
    parser.add_argument("--type", choices=["conda", "venv", "dirs-only"], default="venv", 
                        help="Type of environment to create")
    args = parser.parse_args()

    print("="*40)
    print(" PHARMAGEN SETUP")
    print("="*40)

    check_python_version()
    create_scaffolding()

    if args.type == "conda":
        setup_conda()
    elif args.type == "venv":
        setup_venv()
    
    # Create flag file
    (PROJECT_ROOT / "src/cfg/venv_setup_true").touch()

if __name__ == "__main__":
    main()
