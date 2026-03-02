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
data_handler.py

Implements efficient data loading, cleaning, and preprocessing pipelines.
Designed to handle heterogeneous biological data (scalars, multi-labels).
"""
from collections.abc import Iterable
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch
import torch.utils._typing_utils
from torch.utils.data import Dataset
from typing_extensions import override

logger = logging.getLogger(__name__)
RE_SPLITTERS = re.compile(r"[,;|/]+")


# =============================================================================
# 0. CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class DataConfig:
    """
    Configuration for Data Loading and Processing.
    """

    dataset_path: Path
    feature_cols: list[str]
    target_cols: list[str]
    multi_label_cols: list[str]
    stratify_col: str | None = None
    num_workers: int = 4
    pin_memory: bool = True


# =============================================================================
# 1. PREPROCESSING ENGINE
# =============================================================================


class PGenProcessor(BaseEstimator, TransformerMixin):
    """
    Stateful processor that converts Pandas DataFrame -> Dictionary of Tensors.

    Decouples Pandas overhead from the training loop by pre-computing
    encodings and transforming data into pure PyTorch tensors before
    Dataset instantiation.
    """

    def __init__(self, config: dict, multi_label_cols: list[str] | None = None):
        self.cfg = config
        self.encoders: dict[str, Any] = {}
        self.multi_label_cols = {c.lower() for c in (multi_label_cols or [])}

        # Identify columns
        all_cols = config.get("features", []) + config.get("targets", [])
        self.target_cols = [
            c.lower() for c in config.get("targets", [])
            if c.lower() not in self.multi_label_cols
        ]
        self.scalar_cols = [
            c.lower()
            for c in all_cols
            if c.lower() not in self.multi_label_cols and c.lower() not in self.target_cols
        ]

        self.unknown_token = "__UNKNOWN__"

    def fit(self, df: pd.DataFrame, y=None):
        logger.info("Fitting feature encoders...")

        # Fit Scalars (LabelEncoding)
        for col in self.scalar_cols + self.target_cols:
            if col not in df.columns:
                continue

            uniques = set(df[col].dropna().unique())
            uniques.add(self.unknown_token)

            enc = LabelEncoder()
            enc.fit(sorted(uniques))
            self.encoders[col] = enc

        # Fit Multi-Labels (Binarizer)
        for col in self.multi_label_cols:
            if col not in df.columns:
                continue

            parsed: list[list[str]] = [_split_multilabel_val(v) for v in df[col]]
            enc = MultiLabelBinarizer()
            enc.fit(parsed)
            self.encoders[col] = enc

        return self

    def transform(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        output_payload = {}

        # 1. Transform Scalars
        for col in self.scalar_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            vals = df[col].fillna(self.unknown_token).to_numpy()

            unknown_idx = np.searchsorted(enc.classes_, self.unknown_token)  # noqa

            valid_mask = np.isin(vals, enc.classes_)
            vals[~valid_mask] = self.unknown_token

            encoded_data = enc.transform(vals)
            output_payload[col] = torch.tensor(encoded_data, dtype=torch.long)

        # 2. Transform Targets
        for col in self.target_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            vals = df[col].fillna(self.unknown_token).astype(str).to_numpy()
            valid_mask = np.isin(vals, enc.classes_)
            vals[~valid_mask] = self.unknown_token
            encoded_data = enc.transform(vals)
            output_payload[col] = torch.tensor(encoded_data, dtype=torch.long)

        # 3. Transform Multi-Labels
        for col in self.multi_label_cols:
            if col not in df.columns:
                continue
            enc = self.encoders[col]
            parsed: list[list[str]] = [_split_multilabel_val(v) for v in df[col]]
            mat = enc.transform(parsed).astype(np.float32)
            output_payload[col] = torch.from_numpy(mat)

        return output_payload


# =============================================================================
# 2. MEMORY-OPTIMIZED DATASET
# =============================================================================


class PGenDataset(Dataset):
    """
    Zero-copy PyTorch Dataset.
    """
    def __init__(
        self,
        data_payload: dict[str, torch.Tensor],
        features: list[str],
        targets: list[str],
        multi_label_cols: set[str],
    ):
        self.data = data_payload
        self.features = [f.lower() for f in features]
        self.targets = [t.lower() for t in targets]
        self.ml_cols = {c.lower() for c in multi_label_cols}

        if not self.data:
            raise ValueError("Empty data payload provided to Dataset.")

        first_key = next(iter(self.data))
        self.length = len(self.data[first_key])

    def __len__(self):
        return self.length

    @override
    def __getitem__(self, index):
        x = {k: self.data[k][index] for k in self.features if k in self.data}
        y = {}
        for k in self.targets:
            if k in self.data:
                val = self.data[k][index]
                y[k] = val.float() if k in self.ml_cols else val

        return x, y


# =============================================================================
# 3. DATA LOADING & CLEANING UTILITIES
# =============================================================================
def load_and_clean_dataset(config: DataConfig) -> pd.DataFrame:
    """
    Loads dataset efficiently, validating schema and performing initial cleanup.
    """
    if not config.dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {config.dataset_path}")

    logger.info("Loading dataset from %s...", config.dataset_path.name)

    try:
        df = pd.read_csv(
            config.dataset_path, sep="\t" if config.dataset_path.suffix == ".tsv" else ","
        )
    except ValueError as e:
        logger.error("Read error: %s", e)
        raise

    # Clean content (strip, lower, etc)
    df = clean_dataset_content(df, config.multi_label_cols)

    return df


def clean_dataset_content(
    df: pd.DataFrame,
    multi_label_cols: Iterable[str] | None = None,
    unknown_token: str = "__UNKNOWN__",
) -> pd.DataFrame:
    """
    Performs vectorized cleaning of the DataFrame content:
    1. Normalizes ConsoleIO.print_headers (lowercase, strip).
    2. Fills NaNs with unknown_tConsoleIO.print_successen.
    3. Converts all cells to lowercase string.
    4. Normalizes multi-label delimiters to pipes '|'.

    Args:
        df: Input DataFrame.
        multi_label_cols: List of column names that contain multi-label data.
        unknown_tConsoleIO.print_successen: TConsoleIO.print_successen to use for missing values.

    Returns:
        Cleaned DataFrame.
    """
    # 1. Normalize ConsoleIO.print_headers
    df.columns = df.columns.str.lower().str.strip()

    # 2. Vectorized String Cleaning
    multi_label_set = {c.lower() for c in (multi_label_cols or [])}

    for col in df.columns:
        series = df[col].fillna(unknown_token).astype(str).str.strip().str.lower()

        if col in multi_label_set:
            # Normalize delimiters to pipe '|' for multi-label consistency (vectorized)
            # Use the pattern string for pandas str.replace
            df[col] = series.str.replace(r"[,;|]+\s", "|", regex=True)
        else:
            df[col] = series

    return df

def _split_multilabel_val(val: Any) -> list[str]:
    """
    Splits a single multi-label value into a sorted list of clean label strings.

    Handles strings, lists, tuples and arrays. Applies RE_SPLITTERS so that
    all delimiter variants (comma, semicolon, pipe, slash) are treated uniformly
    regardless of whether clean_dataset_content has already been called.

    This is the single source of truth for multi-label parsing — used by
    serialize_multilabel, PGenProcessor.fit and PGenProcessor.transform.
    """
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, str):
        parts = [s.strip() for s in RE_SPLITTERS.split(val) if s.strip()]
    elif isinstance(val, (list, tuple, np.ndarray)):
        # Already split upstream — flatten and re-clean each element
        parts = [s.strip() for item in val for s in RE_SPLITTERS.split(str(item)) if s.strip()]
    else:
        parts = [str(val).strip()]
    # sorted() for deterministic order (reproducible vocab indices)
    return sorted(set(parts))

#################################################################################################

if __name__ == "__main__":
    import argparse
    from collections import Counter
    from pathlib import Path
    import sys

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from src.interface.ui import ConsoleIO

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Diagnóstico del pipeline de datos de Pharmagen.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/processed/final_training_data.tsv"),
        help="Ruta al dataset (CSV o TSV). Default: data/processed/final_training_data.tsv",
    )
    parser.add_argument(
        "--features", "-f",
        nargs="+",
        default=["atc", "drug", "gene_symbol", "variant_normalized", "genotype",
                 "previous_condition_term"],
        help="Columnas de features.",
    )
    parser.add_argument(
        "--targets", "-t",
        nargs="+",
        default=["phenotype_outcome", "effect_direction", "effect_type"],
        help="Columnas target.",
    )
    parser.add_argument(
        "--multi-label", "-ml",
        nargs="*",
        default=[],
        dest="multi_label",
        help="Subconjunto de targets que son multi-label (separados por pipe).",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fracción de validación para el split. Default: 0.2",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria. Default: 42",
    )
    args = parser.parse_args()



    # ------------------------------------------------------------------
    # Helpers de consola
    # ------------------------------------------------------------------
    W = 70  # ancho de línea


    def section(title: str):
        print(f"\n  {'─' * (W - 4)}")
        print(f"  ▶  {title}")
        print(f"  {'─' * (W - 4)}")

    def row(label: str, value, width: int = 30):
        print(f"    {label:<{width}} {value}")

    # ------------------------------------------------------------------
    # BLOQUE 0 — Carga y limpieza básica
    # ------------------------------------------------------------------
    ConsoleIO.print_header("PHARMAGEN · DIAGNÓSTICO DE DATA_HANDLER")

    if not args.input.exists():
        ConsoleIO.print_error(f"Archivo no encontrado: {args.input}")
        sys.exit(1)

    config = DataConfig(
        dataset_path=args.input,
        feature_cols=args.features,
        target_cols=args.targets,
        multi_label_cols=args.multi_label,
    )

    print(f"\n  Cargando: {args.input}")
    df_raw = load_and_clean_dataset(config)

    section("0 · Resumen del dataset cargado")
    row("Archivo",        args.input.name)
    row("Filas totales",  len(df_raw))
    row("Columnas",       len(df_raw.columns))
    row("Features",       args.features)
    row("Targets",        args.targets)
    row("Multi-label",    args.multi_label or "ninguno")

    # Columnas esperadas vs presentes
    all_expected = set(args.features) | set(args.targets)
    missing_cols = all_expected - set(df_raw.columns)
    extra_cols = set(df_raw.columns) - all_expected

    if missing_cols:
        ConsoleIO.print_warning(f"Columnas esperadas AUSENTES: {sorted(missing_cols)}")
    else:
        ConsoleIO.print_success("Todas las columnas esperadas están presentes.")

    if extra_cols:
        row("Columnas extra (ignoradas)", sorted(extra_cols))

    # ------------------------------------------------------------------
    # BLOQUE 1 — Separación de etiquetas multi-label
    # ------------------------------------------------------------------
    ConsoleIO.print_header("BLOQUE 1 · PARSING MULTI-LABEL")

    if not args.multi_label:
        ConsoleIO.print_info("\n No se definieron columnas multi-label. Saltando bloque 1.")
    else:
        for col in args.multi_label:
            if col not in df_raw.columns:
                ConsoleIO.print_warning(f"Columna multi-label '{col}' no encontrada en el dataset.")
                continue

            section(f"Columna: {col}")
            series = df_raw[col].astype(str)

            # Conteo de separadores detectados
            n_pipe      = series.str.contains(r"\|", regex=True).sum()
            n_comma     = series.str.contains(r",", regex=False).sum()
            n_semicolon = series.str.contains(r";", regex=False).sum()
            n_unknown   = (series == "__unknown__").sum()
            n_empty     = (series.str.strip() == "").sum()

            row("Filas totales",                   len(series))
            row("Con separador pipe '|'",           n_pipe)
            row("Con coma ',' (pre-normalización)", n_comma)
            row("Con punto y coma ';'",             n_semicolon)
            row("Valor __UNKNOWN__",                n_unknown)
            row("Vacíos tras limpieza",             n_empty)

            # Cardinalidad tras el split
            parsed = series.str.split("|").apply(
                lambda x: [i.strip() for i in x if i.strip() and i.strip() != "__unknown__"]
            )
            all_labels = [label for sublist in parsed for label in sublist]
            label_counts = Counter(all_labels)
            n_unique_labels = len(label_counts)

            ConsoleIO.print_divider()
            row("Etiquetas únicas detectadas", n_unique_labels)
            row("Instancias totales de etiqueta", len(all_labels))

            # Distribución de cantidad de etiquetas por muestra
            label_per_sample = parsed.apply(len)
            row("Etiquetas/muestra — media",  f"{label_per_sample.mean():.2f}")
            row("Etiquetas/muestra — máx",    label_per_sample.max())
            row("Muestras con 0 etiquetas",   (label_per_sample == 0).sum())
            row("Muestras con 1 etiqueta",    (label_per_sample == 1).sum())
            row("Muestras con ≥2 etiquetas",  (label_per_sample >= 2).sum())

            # Top-10 etiquetas más frecuentes
            ConsoleIO.print_divider()
            print(f"    {'Etiqueta':<35} {'Count':>8}  {'%':>6}")
            print(f"    {'─'*35} {'─'*8}  {'─'*6}")
            for label, count in label_counts.most_common(10):
                pct = count / len(all_labels) * 100
                print(f"    {label:<35} {count:>8}  {pct:>5.1f}%")

            # Etiquetas raras (< 1% del total)
            rare = [(lab, c) for lab, c in label_counts.items() if c / len(all_labels) < 0.01]
            if rare:
                ConsoleIO.print_warning(f"{len(rare)} etiquetas con < 1% de frecuencia (posible ruido).")
            else:
                ConsoleIO.print_success("Ninguna etiqueta extremadamente rara (< 1%).")

    # ------------------------------------------------------------------
    # BLOQUE 2 — Distribución de clases por target
    # ------------------------------------------------------------------
    ConsoleIO.print_header("BLOQUE 2 · DISTRIBUCIÓN DE CLASES POR TARGET")

    for col in args.targets:
        if col not in df_raw.columns:
            ConsoleIO.print_warning(f"Target '{col}' no encontrado. Saltando.")
            continue

        section(f"Target: {col}  {'[MULTI-LABEL]' if col in args.multi_label else '[MULTI-CLASS]'}")
        series = df_raw[col].astype(str)
        value_counts = series.value_counts()
        n_classes = len(value_counts)
        total = len(series)

        row("Clases únicas",     n_classes)
        row("Clase más frecuente",  f"{value_counts.index[0]} ({value_counts.iloc[0]})")
        row("Clase menos frecuente", f"{value_counts.index[-1]} ({value_counts.iloc[-1]})")

        # Ratio de imbalanceo (max / min)
        imbalance_ratio = value_counts.iloc[0] / value_counts.iloc[-1]
        row("Ratio imbalanceo (max/min)", f"{imbalance_ratio:.1f}x")

        if imbalance_ratio > 10:
            ConsoleIO.print_warning(f"Desbalanceo severo ({imbalance_ratio:.1f}x). Considera focal/ASL loss o oversampling.")
        elif imbalance_ratio > 3:
            ConsoleIO.print_warning(f"Desbalanceo moderado ({imbalance_ratio:.1f}x). Monitorizar métricas por clase.")
        else:
            ConsoleIO.print_success(f"Distribución razonablemente balanceada ({imbalance_ratio:.1f}x).")

        ConsoleIO.print_divider()
        # Tabla de distribución (top 15 para no saturar consola)
        max_display = 15
        print(f"    {'Clase':<35} {'Count':>8}  {'%':>6}  {'Bar'}")
        print(f"    {'─'*35} {'─'*8}  {'─'*6}  {'─'*20}")
        for cls, count in value_counts.head(max_display).items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)  # escala: 50% = 25 chars
            print(f"    {cls!s:<35} {count:>8}  {pct:>5.1f}%  {bar}")
        if n_classes > max_display:
            print(f"    ... y {n_classes - max_display} clases más (no mostradas)")

    # ------------------------------------------------------------------
    # BLOQUE 3 — TOKENS OOV (__UNKNOWN__) tras el encode
    # ------------------------------------------------------------------
    ConsoleIO.print_header("BLOQUE 3 · TOKENS OOV TRAS ENCODE (TRAIN → VAL)")

    # Necesitamos hacer el split y fitear el procesador igual que en pipeline.py
    stratify_col = config.stratify_col
    stratify_vals = df_raw[stratify_col] if stratify_col and stratify_col in df_raw.columns else None

    try:
        train_df, val_df = train_test_split(
            df_raw,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=stratify_vals,
        )
    except ValueError as e:
        ConsoleIO.print_warning(f"Estratificación fallida ({e}). Usando split aleatorio simple.")
        train_df, val_df = train_test_split(df_raw, test_size=args.val_size, random_state=args.seed)

    processor = PGenProcessor(
        config={"features": args.features, "targets": args.targets},
        multi_label_cols=args.multi_label,
    )
    processor.fit(train_df)

    section("Vocabularios aprendidos en TRAIN")
    for col, enc in processor.encoders.items():
        if hasattr(enc, "classes_"):
            n_classes = len(enc.classes_)
            has_unknown = "__unknown__" in [c.lower() for c in enc.classes_]
            flag = "✅" if has_unknown else "⚠️ "
            print(f"    {flag}  {col:<35} vocab={n_classes:>5}  __UNKNOWN__={'sí' if has_unknown else 'NO'}")

    section("OOV en conjunto de VALIDACIÓN")

    oov_report = {}
    for col in args.features:
        if col not in processor.encoders or col not in val_df.columns:
            continue
        enc = processor.encoders[col]
        vals = val_df[col].fillna("__UNKNOWN__").astype(str).to_numpy()
        oov_mask = ~np.isin(vals, enc.classes_)
        n_oov = oov_mask.sum()
        pct_oov = n_oov / len(vals) * 100
        oov_report[col] = (n_oov, pct_oov)

    any_oov = False
    print(f"\n    {'Feature':<35} {'OOV count':>10}  {'OOV %':>7}")
    print(f"    {'─'*35} {'─'*10}  {'─'*7}")
    for col, (n_oov, pct_oov) in oov_report.items():
        flag = "⚠️ " if pct_oov > 5 else "   "
        print(f"    {flag}{col:<33} {n_oov:>10}  {pct_oov:>6.2f}%")
        if pct_oov > 5:
            any_oov = True

    if any_oov:
        print()
        ConsoleIO.print_warning("Features con > 5% OOV — vocabulario de train puede ser insuficiente.")
        ConsoleIO.print_warning("Considera: más datos, vocab mínimo por clase, o suavizado de OOV.")
    else:
        print()
        ConsoleIO.print_success("Todos los features tienen < 5% OOV en validación.")

    # Mismo análisis para targets (OOV en targets = problema grave)
    section("OOV en TARGETS de validación (debe ser 0%)")
    target_oov_found = False
    for col in args.targets:
        if col in args.multi_label or col not in processor.encoders or col not in val_df.columns:
            continue
        enc = processor.encoders[col]
        vals = val_df[col].fillna("__UNKNOWN__").astype(str).to_numpy()
        oov_mask = ~np.isin(vals, enc.classes_)
        n_oov = oov_mask.sum()
        pct_oov = n_oov / len(vals) * 100
        flag = "❌" if n_oov > 0 else "✅"
        print(f"    {flag}  {col:<35} OOV={n_oov}  ({pct_oov:.2f}%)")
        if n_oov > 0:
            target_oov_found = True
            unique_oov = set(vals[oov_mask])
            ConsoleIO.print_warning(f"Clases no vistas en train para '{col}': {sorted(unique_oov)[:5]}")

    if not target_oov_found:
        ConsoleIO.print_success("Ningún target tiene clases OOV en validación.")

    # ------------------------------------------------------------------
    # BLOQUE 4 — Integridad del split train/val
    # ------------------------------------------------------------------
    ConsoleIO.print_header("BLOQUE 4 · INTEGRIDAD DEL SPLIT TRAIN / VAL")

    section("Tamaños del split")
    row("Total filas",      len(df_raw))
    row("Train",           f"{len(train_df)}  ({len(train_df)/len(df_raw)*100:.1f}%)")
    row("Val",             f"{len(val_df)}  ({len(val_df)/len(df_raw)*100:.1f}%)")
    row("Filas objetivo val", f"{int(len(df_raw) * args.val_size)}  (val_size={args.val_size})")

    section("Fuga de datos (data leakage)")
    # Comprobar solapamiento de índices
    train_idx = set(train_df.index)
    val_idx = set(val_df.index)
    overlap = train_idx & val_idx
    if overlap:
        ConsoleIO.print_warning(f"FUGA DETECTADA: {len(overlap)} índices en train Y val. ¡Revisar split!")
    else:
        ConsoleIO.print_success("Sin solapamiento de índices entre train y val.")

    # Comprobar solapamiento de filas (duplicados exactos que pueden aparecer en ambos splits)
    # Muestreo para no saturar memoria en datasets grandes
    sample_size = min(5000, len(train_df), len(val_df))
    train_sample = train_df.sample(sample_size, random_state=args.seed)
    val_sample = val_df.sample(sample_size, random_state=args.seed)

    train_tuples = set(map(tuple, train_sample[args.features].values.tolist()))
    val_tuples   = set(map(tuple, val_sample[args.features].values.tolist()))
    shared_rows = train_tuples & val_tuples

    if shared_rows:
        ConsoleIO.print_warning(
            f"{len(shared_rows)} filas de features idénticas en train y val "
            f"(muestreo de {sample_size}). Puede indicar duplicados en el dataset original."
        )
    else:
        ConsoleIO.print_success(f"Sin filas de features idénticas entre splits (muestreo de {sample_size}).")

    section("Distribución por target en train vs val")
    for col in args.targets:
        if col in args.multi_label or col not in df_raw.columns:
            continue

        train_dist = train_df[col].value_counts(normalize=True).sort_index()
        val_dist   = val_df[col].value_counts(normalize=True).sort_index()

        # Alinear índices
        all_classes = train_dist.index.union(val_dist.index)
        train_dist = train_dist.reindex(all_classes, fill_value=0.0)
        val_dist   = val_dist.reindex(all_classes, fill_value=0.0)

        # Divergencia máxima absoluta entre distribuciones
        max_drift = (train_dist - val_dist).abs().max()

        flag = "⚠️ " if max_drift > 0.05 else "✅ "
        print(f"\n    {flag} {col}  (drift máximo: {max_drift:.3f})")

        if max_drift > 0.05:
            ConsoleIO.print_warning("Drift > 5% en alguna clase. El split no es perfectamente estratificado.")
            # Mostrar las clases con mayor drift
            diffs = (train_dist - val_dist).abs().sort_values(ascending=False)
            print(f"\n    {'Clase':<30} {'Train%':>8}  {'Val%':>8}  {'|Δ|':>6}")
            print(f"    {'─'*30} {'─'*8}  {'─'*8}  {'─'*6}")
            for cls in diffs.head(5).index:
                print(
                    f"    {cls!s:<30} "
                    f"{train_dist[cls]:>7.1%}  "
                    f"{val_dist[cls]:>7.1%}  "
                    f"{diffs[cls]:>5.3f}"
                )

    # ------------------------------------------------------------------
    # RESUMEN FINAL
    # ------------------------------------------------------------------
    ConsoleIO.print_header("RESUMEN FINAL")

    print(f"""
  Dataset:     {args.input.name}
  Filas:       {len(df_raw)}  →  Train {len(train_df)} / Val {len(val_df)}
  Features:    {len(args.features)}  |  Targets: {len(args.targets)}
  Multi-label: {args.multi_label or 'ninguno'}
  Encoders:    {len(processor.encoders)} ajustados

  Recuerda antes de lanzar el HPO:
    · OOV > 5% en features   → ampliar vocab o añadir más datos
    · OOV > 0% en targets     → revisar limpieza / split
    · Imbalanceo > 10x        → focal_gamma alto o ASL agresivo
    · Drift > 5% en split     → usar stratify_col en train_test_split
    · Etiquetas raras < 1%    → consolidar clases o umbral mínimo
""")
    print("═" * W)
