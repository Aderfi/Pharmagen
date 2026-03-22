# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025  Adrim Hamed Outmani
import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.interface.ui import ConsoleIO


def generate_eval_report(
    all_preds: dict[str, list],
    all_targets: dict[str, list],
    multi_label_cols: set[str],
    output_dir: Path,
    study_name: str,
    class_names: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """
    Computes and exports F1-Score, Balanced Accuracy and Confusion Matrices
    for the best Optuna trial, handling multi-label and multi-class targets separately.

    Args:
        all_preds: {target_col -> list of predictions} from PGenTrainer.predict()
        all_targets: {target_col -> list of ground truths} from PGenTrainer.predict()
        multi_label_cols: set of target columns that are multi-label
        output_dir: base reports directory
        study_name: used for file naming
        class_names: optional {target_col -> [label_name, ...]} from processor.encoders[col].classes_
                     If provided, metrics dicts use label names as keys instead of numeric indices.

    Returns:
        metrics_report: dict with all computed metrics (also saved as JSON)
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            balanced_accuracy_score,
            confusion_matrix,
            f1_score,
        )
    except ImportError as e:
        ConsoleIO.print_error(f"Missing dependency for eval report: {e}")
        ConsoleIO.print_info("Install with: pip install scikit-learn matplotlib numpy")
        return {}

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_report: dict[str, Any] = {}

    ConsoleIO.print_divider("=", 40)
    ConsoleIO.print_info(f"Evaluation Metrics — Best Trial ({study_name})")
    ConsoleIO.print_divider("=", 40)

    for target_col, preds in all_preds.items():
        if target_col not in all_targets or len(preds) == 0:
            continue

        y_pred = np.array(preds)
        y_true = np.array(all_targets[target_col])
        is_ml = target_col in multi_label_cols

        col_metrics: dict[str, Any] = {"type": "multi-label" if is_ml else "multi-class"}

        # Resolve label names for this column (fallback to "label_0", "label_1", ...)
        col_class_names: list[str] = (
            list(class_names[target_col])
            if class_names and target_col in class_names
            else []
        )

        if is_ml:
            # --- Multi-label ---
            # F1 per label + macro average
            f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            f1_per_label_values = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

            # Balanced accuracy per label, then averaged
            n_labels = y_true.shape[1] if y_true.ndim > 1 else 1
            bal_acc_per_label_values: list[float] = []
            for i in range(n_labels):
                col_true = y_true[:, i] if y_true.ndim > 1 else y_true
                col_pred = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                if len(np.unique(col_true)) > 1:
                    bal_acc_per_label_values.append(float(balanced_accuracy_score(col_true, col_pred)))
                else:
                    bal_acc_per_label_values.append(float("nan"))
            bal_acc_macro = float(np.nanmean(bal_acc_per_label_values)) if bal_acc_per_label_values else 0.0

            # Build label name list — pad with "label_N" if encoder had fewer classes than predictions
            label_names: list[str] = [
                col_class_names[i] if i < len(col_class_names) else f"label_{i}"
                for i in range(n_labels)
            ]

            col_metrics.update({
                "f1_macro": round(f1_macro, 4),
                "f1_per_label": {
                    label_names[i]: round(v, 4) for i, v in enumerate(f1_per_label_values)
                },
                "balanced_accuracy_macro": round(bal_acc_macro, 4),
                "balanced_accuracy_per_label": {
                    label_names[i]: round(v, 4) for i, v in enumerate(bal_acc_per_label_values)
                },
            })

            ConsoleIO.print_info(f"[{target_col}] (multi-label)")
            ConsoleIO.print_info(f"  F1 Macro:           {f1_macro:.4f}")
            ConsoleIO.print_info(f"  Balanced Acc Macro: {bal_acc_macro:.4f}")
            for lname, fval in col_metrics["f1_per_label"].items():
                ConsoleIO.print_info(f"    {lname:<30} F1={fval:.4f}  BalAcc={col_metrics['balanced_accuracy_per_label'][lname]:.4f}")

            # Confusion matrix per label
            n_plots = n_labels
            ncols = min(4, n_plots)
            nrows = (n_plots + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            axes_flat = np.array(axes).flatten() if n_plots > 1 else [axes]

            for i in range(n_plots):
                col_true = y_true[:, i] if y_true.ndim > 1 else y_true
                col_pred = y_pred[:, i] if y_pred.ndim > 1 else y_pred
                cm = confusion_matrix(col_true, col_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Absent", "Present"])
                disp.plot(ax=axes_flat[i], colorbar=False)
                axes_flat[i].set_title(label_names[i])

            for j in range(n_plots, len(axes_flat)):
                axes_flat[j].set_visible(False)

            fig.suptitle(f"Confusion Matrices — {target_col}", fontsize=14)
            plt.tight_layout()
            fig_path = figures_dir / f"{study_name}_cm_{target_col}.png"
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        else:
            # --- Multi-class ---
            f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            bal_acc = float(balanced_accuracy_score(y_true, y_pred))

            # F1 per class with names
            f1_per_class_values = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            mc_class_names: list[str] = [
                col_class_names[c] if c < len(col_class_names) else f"class_{c}"
                for c in unique_classes
            ]

            col_metrics.update({
                "f1_macro": round(f1_macro, 4),
                "f1_weighted": round(f1_weighted, 4),
                "f1_per_class": {
                    mc_class_names[i]: round(v, 4) for i, v in enumerate(f1_per_class_values)
                },
                "balanced_accuracy": round(bal_acc, 4),
            })

            ConsoleIO.print_info(f"[{target_col}] (multi-class)")
            ConsoleIO.print_info(f"  F1 Macro:        {f1_macro:.4f}")
            ConsoleIO.print_info(f"  F1 Weighted:     {f1_weighted:.4f}")
            ConsoleIO.print_info(f"  Balanced Acc:    {bal_acc:.4f}")
            for cname, fval in col_metrics["f1_per_class"].items():
                ConsoleIO.print_info(f"    {cname:<30} F1={fval:.4f}")

            # Confusion matrix with class names
            cm = confusion_matrix(y_true, y_pred)
            n_classes = len(unique_classes)
            fig, ax = plt.subplots(figsize=(max(6, n_classes), max(5, n_classes)))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mc_class_names)
            disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
            ax.set_title(f"Confusion Matrix — {target_col}")
            plt.tight_layout()
            fig_path = figures_dir / f"{study_name}_cm_{target_col}.png"
            plt.savefig(fig_path, dpi=120, bbox_inches="tight")
            plt.close(fig)

        metrics_report[target_col] = col_metrics

    ConsoleIO.print_divider("=", 40)

    # Save JSON
    json_path = output_dir / f"{study_name}_eval_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=4)
    ConsoleIO.print_info(f"Eval metrics saved → {json_path}")

    return metrics_report


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
    row_list = []
    for row in df.iloc[::-1].to_dict('records'):
        acc_cells = "".join(f"<td>{row[c]:.2%}</td>" for c in acc_cols)
        row_list.append(f"""
        <tr>
            <td>{int(row["epoch"])}</td>
            <td>{row["train_loss"]:.4f}</td>
            <td>{row["val_loss"]:.4f}</td>
            {acc_cells}
        </tr>
        """)
    return "".join(row_list)
