# src/utils/reporting.py
# Pharmagen - Reporting Module
import base64
from io import BytesIO
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
    except ImportError as e:
        ConsoleIO.print_error(f" Could not generate HTML report: {e}")
        ConsoleIO.print_info("-- Missing dependencies. [Matplotlib]")
    except Exception as e:
        ConsoleIO.print_error(f" Failed to generate HTML report: {e}")


def _create_html_report(history: list[dict[str, Any]], output_dir: Path, model_name: str):
    import matplotlib.pyplot as plt

    df = pd.DataFrame(history)

    # Generate Plots
    imgs = {}

    # A. Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", linewidth=2, linestyle="--")
    plt.title(f"{model_name} - Loss Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    imgs["loss"] = _fig_to_base64(plt)
    plt.close()

    # B. Accuracy Plot (Multi-task)
    acc_cols = [c for c in df.columns if c.startswith("val_acc_")]

    if acc_cols:
        plt.figure(figsize=(10, 6))
        for col in acc_cols:
            label = col.replace("val_acc_", "").replace("_", " ").title()
            plt.plot(df["epoch"], df[col], label=label, linewidth=2)

        plt.title(f"{model_name} - Accuracy Metrics (Validation)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        imgs["acc"] = _fig_to_base64(plt)
        plt.close()

    # C. Build HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pharmagen Report - {model_name}</title>
        <style>
            :root {{ --primary: #2563eb; --bg: #f8fafc; --card: #ffffff; --text: #1e293b; }}
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                    background: var(--bg); color: var(--text); margin: 0; padding: 20px; line-height: 1.5; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
            .header h1 {{ margin: 0; color: var(--primary); font-size: 1.5rem; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-top: 15px; }}
            .stat-box {{ background: #eff6ff; padding: 10px; border-radius: 8px; text-align: center; }}
            .stat-value {{ display: block; font-size: 1.25rem; font-weight: bold; color: var(--primary); }}
            .stat-label {{ font-size: 0.875rem; color: #64748b; }}

            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .card {{ background: var(--card); padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            h2 {{ font-size: 1.2rem; margin-top: 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 10px; }}
            img {{ width: 100%; height: auto; display: block; }}

            table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
            th {{ background: #f1f5f9; text-align: left; padding: 12px; font-weight: 600; }}
            td {{ padding: 12px; border-bottom: 1px solid #e2e8f0; }}
            tr:hover {{ background: #f8fafc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>💊 Pharmagen Training Report: {model_name}</h1>
                <div class="stats">
                    <div class="stat-box">
                        <span class="stat-value">{len(df)}</span>
                        <span class="stat-label">Total Epochs</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-value">{df["val_loss"].min():.4f}</span>
                        <span class="stat-label">Best Val Loss</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-value">{df["val_acc_macro"].max():.2%}</span>
                        <span class="stat-label">Best Macro Accuracy</span>
                    </div>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>📉 Loss History</h2>
                    <img src="data:image/png;base64,{imgs.get("loss", "")}" alt="Loss Plot" />
                </div>
                {f'<div class="card"><h2>🎯 Accuracy History</h2><img src="data:image/png;base64,{imgs["acc"]}" alt="Acc Plot" /></div>' if "acc" in imgs else ""}
            </div>

            <div class="card">
                <h2>📋 Detailed Logs</h2>
                <div style="overflow-x: auto;">
                    <table>
                        <thead>
                            <tr>
                                <th>Epoch</th>
                                <th>Train Loss</th>
                                <th>Val Loss</th>
                                {"".join(f"<th>{c.replace('val_acc_', '')}</th>" for c in acc_cols)} 
                            </tr>
                        </thead>
                        <tbody>
                            {_generate_table_rows(df, acc_cols)}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </body>
    </html>
    """ # noqa

    with open(output_dir / f"{model_name}_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)


def _fig_to_base64(plt_obj):
    buf = BytesIO()
    plt_obj.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _generate_table_rows(df, acc_cols):
    rows = ""
    for _, row in df.iloc[::-1].iterrows():
        acc_cells = "".join(f"<td>{row[c]:.2%}</td>" for c in acc_cols)
        rows += f"""
        <tr>
            <td>{int(row["epoch"])}</td>
            <td>{row["train_loss"]:.4f}</td>
            <td>{row["val_loss"]:.4f}</td>
            {acc_cells}
        </tr>
        """
    return rows
