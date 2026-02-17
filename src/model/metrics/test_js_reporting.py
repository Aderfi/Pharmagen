# src/utils/reporting.py
# Pharmagen - Reporting Module
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.interface.ui import ConsoleIO


def generate_training_report(history: list[dict[str, Any]], output_dir: Path, model_name: str):
    """
    Generates HTML and JSON reports from training history.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save JSON
    json_path = output_dir / f"{model_name}_history.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    # 2. Generate HTML
    try:
        _create_html_report(history, output_dir, model_name)
    except Exception as e:
        ConsoleIO.print_error(f" Failed to generate HTML report: {e}")


def _create_html_report(history: list[dict[str, Any]], output_dir: Path, model_name: str):
    df = pd.DataFrame(history)

    # Detect accuracy columns dynamically (any col starting with val_acc_)
    acc_cols = [c for c in df.columns if c.startswith("val_acc_")]

    # Build the data payload for the JS template
    report_data = {
        "model_name": model_name,
        "acc_cols": acc_cols,
        "history": history,
    }

    # Read template and inject data
    template_path = Path(__file__).parent / "rframework.html"
    with open(template_path, encoding="utf-8") as f:
        html = f.read()

    html = html.replace("__MODEL_NAME__", model_name)
    html = html.replace("__REPORT_DATA_JSON__", json.dumps(report_data, ensure_ascii=False))

    output_path = output_dir / f"{model_name}_report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
