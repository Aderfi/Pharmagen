"""
build_dataset.py
================
Pipeline de construcción del dataset tabular para PharmGKB → DeepFM.

Carga, filtra y transforma las anotaciones de variantes de PharmGKB en un
fichero TSV listo para entrenamiento con características categóricas.

Uso:
    python build_dataset.py --input-dir data/raw/variantAnnotations \
                            --output data/processed/final_pharmagen_dataset.tsv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Final

import numpy as np
import pandas as pd

from src.interface.ui import ConsoleIO

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
UNKNOWN: Final[str] = "__UNKNOWN__"
UNKNOWN_GENE: Final[str] = "__INTRONIC__"
UNKNOWN_ALLELE: Final[str] = "__NO_ALLELE__"
UNKNOWN_DIRECTION: Final[str] = "__UNDETERMINED__"

# Valores que se tratarán como "desconocido" durante la limpieza
_NULL_LIKE: Final[frozenset[str]] = frozenset({"nan", "none", "", "unknown"})

# Archivos de anotaciones esperados
_ANNOTATION_FILES: Final[tuple[str, ...]] = (
    "var_drug_ann.tsv",
    "var_fa_ann.tsv",
    "var_pheno_ann.tsv",
)

# Mapeo de columnas originales → nombres internos
_COLUMN_MAPPING: Final[dict[str, str]] = {
    "Variant/Haplotypes": "Variant_Haplotype",
    "Gene": "Gene",
    "Phenotype Category": "Phenotype_Outcome",
    "Direction of effect": "Effect_Direction",
    "Metabolizer types": "Metabolizer_Status",
    #"Side effect/efficacy/other": "Effect_type",
    #"PD/PK terms": "Effect_type",
    #"Functional terms": "Effect_type",
    "Phenotype": "Subtype",  # Columna vacía en algunos archivos, se rellena con UNKNOWN
    "Population Phenotypes or diseases": "Previous_Condition_Term",  # Se prioriza este nombre para la columna de condiciones previas
}

_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "Drug",
    "Gene",
    "Variant_Haplotype",
    "Allele",
    "Phenotype_Outcome",
    "Effect_Direction",
    "Effect_type",
    "Subtype",
    "Previous_Condition_Term",
    "Metabolizer_Status",
)

_FINAL_COLUMNS: Final[tuple[str, ...]] = (
    "Drug",
    "Variant_Haplotype",
    "Gene",
    "Allele",
    "Genalle",
    "Phenotype_Outcome",
    "Effect_Direction",
    "Effect_type",
    "Subtype",
    "Previous_Condition_Term",
    "Metabolizer_Status",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_str_series(series: pd.Series) -> pd.Series:
    """Normaliza una Serie de strings: strip, lower y reemplaza nulos."""
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.where(~cleaned.isin(_NULL_LIKE), other=UNKNOWN)


def _get_lowercase_col_map(df: pd.DataFrame) -> dict[str, str]:
    """Devuelve un dict {nombre_original: nombre_en_minúsculas_stripped}."""
    return {c: c.lower().strip() for c in df.columns}


# ---------------------------------------------------------------------------
# Etapas del pipeline
# ---------------------------------------------------------------------------

def load_and_filter(filepath: Path) -> pd.DataFrame:
    """
    Carga un TSV de anotaciones y aplica los filtros de exclusión.

    Filtra filas donde:
    - 'significance' == 'no'
    - 'is/is not associated' == 'not associated with'

    Returns
    -------
    pd.DataFrame
        DataFrame filtrado, o vacío si hay error de lectura.
    """
    ConsoleIO.print_info(f"Cargando {filepath.name}...")
    try:
        # dtype=str evita inferencias costosas con low_memory=False en pandas ≥ 2
        df = pd.read_csv(filepath, sep="\t", dtype=str)
    except OSError as exc:
        ConsoleIO.print_error(f"Error al leer {filepath.name}: {exc}")
        return pd.DataFrame()

    lower_cols: dict[str, str] = _get_lowercase_col_map(df)
    # Vista con nombres en minúsculas para búsqueda segura (sin copia extra)
    df_lower = df.rename(columns=lower_cols)

    mask = pd.Series(True, index=df.index)

    if "significance" in df_lower.columns:
        mask &= df_lower["significance"].str.lower().str.strip() != "no"

    if "is/is not associated" in df_lower.columns:
        mask &= (
            df_lower["is/is not associated"].str.lower().str.strip()
            != "not associated with"
        )

    df_filtered = df.loc[mask].copy()
    logger.info("%s — filas tras filtrado: %d", filepath.name, len(df_filtered))
    ConsoleIO.print_info(f"{filepath.name} — filas tras filtrado: {len(df_filtered)}")
    return df_filtered


def normalize_drugs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expande filas con múltiples fármacos separados por coma en la columna 'Drug(s)'.

    Crea una nueva columna 'Drug' con un único fármaco por fila.
    Las filas con valores nulos o desconocidos son eliminadas.

    Returns
    -------
    pd.DataFrame
        DataFrame con columna 'Drug' normalizada y una entrada por fármaco.
    """
    if "Drug(s)" not in df.columns:
        logger.warning("Columna 'Drug(s)' no encontrada; se omite normalización de fármacos.")
        df = df.copy()
        df["Drug"] = UNKNOWN
        return df

    df = df.copy()
    df["Drug"] = df["Drug(s)"].astype(str).str.split(r",\s*")
    df = df.explode("Drug")
    df["Drug"] = df["Drug"].str.lower().str.strip()
    df = df[~df["Drug"].isin(_NULL_LIKE)]
    return df


def process_alleles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza la columna de alelos.

    Si no existe la columna 'Alleles', asigna UNKNOWN a todos los registros.

    Returns
    -------
    pd.DataFrame
        DataFrame con columna 'Allele' normalizada.
    """
    df = df.copy()

    if "Alleles" not in df.columns:
        logger.warning("Columna 'Alleles' no encontrada; se asigna '%s'.", UNKNOWN)
        df["Allele"] = UNKNOWN
        return df

    df["Allele"] = (
        df["Alleles"]
        .fillna(UNKNOWN)
        .astype(str)
        .str.strip()
        .replace({"": UNKNOWN, "nan": UNKNOWN})
    )
    return df


def rename_and_fill_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renombra columnas al esquema interno y rellena las ausentes con UNKNOWN.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas renombradas y garantizadas.
    """
    df = df.copy()

    # Asegurar existencia de todas las columnas del mapeo antes de renombrar
    for original_col in _COLUMN_MAPPING:
        if original_col not in df.columns:
            logger.debug("Columna '%s' ausente; se añade con valor '%s'.", original_col, UNKNOWN)
            df[original_col] = np.nan

    df = df.rename(columns=_COLUMN_MAPPING)

    # Normalizar todas las features categóricas
    for col in _CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = _clean_str_series(df[col])
        else:
            df[col] = UNKNOWN

    return df

def process_and_normalize_effect_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza la columna 'Effect_type' combinando varias columnas de origen.

    Si 'Effect_type' ya existe, se limpia y normaliza. Si no, se intenta construir
    a partir de las columnas 'Side effect/efficacy/other', 'PD/PK terms' y
    'Functional terms'. Si ninguna de estas columnas existe, se asigna UNKNOWN.

    Returns
    -------
    pd.DataFrame
        DataFrame con columna 'Effect_type' normalizada.
    """
    df = df.copy()

    if "Effect_type" in df.columns:
        df["Effect_type"] = _clean_str_series(df["Effect_type"])
        return df

    effect_cols = ["Side effect/efficacy/other", "PD/PK terms", "Functional terms"]
    existing_effect_cols = [col for col in effect_cols if col in df.columns]

    if not existing_effect_cols:
        logger.warning("Ninguna columna de efecto encontrada; se asigna '%s' a 'Effect_type'.", UNKNOWN)
        df["Effect_type"] = UNKNOWN
        return df

    # Combinar las columnas existentes en una sola columna 'Effect_type'
    df["Effect_type"] = (
        df[existing_effect_cols]
        .fillna("")
        .agg(lambda x: ", ".join(filter(None, x)), axis=1)
        .str.strip(", ")
    )
    df["Effect_type"] = _clean_str_series(df["Effect_type"])
    return df

def build_genalle(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la feature compuesta 'Genalle' = Gene + '_' + Allele.

    Cuando ambos son UNKNOWN, la feature resultante también es UNKNOWN.

    Returns
    -------
    pd.DataFrame
        DataFrame con columna 'Genalle' añadida.
    """
    df = df.copy()
    both_unknown = (df["Gene"] == UNKNOWN) & (df["Allele"] == UNKNOWN)
    df["Genalle"] = df["Gene"] + "_" + df["Allele"]
    df.loc[both_unknown, "Genalle"] = UNKNOWN
    return df


def select_and_deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona las columnas finales y elimina duplicados exactos.

    Returns
    -------
    pd.DataFrame
        DataFrame deduplicado con solo las columnas necesarias.

    Raises
    ------
    ValueError
        Si alguna columna de _FINAL_COLUMNS no existe en el DataFrame.
    """
    missing = [c for c in _FINAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas ausentes en el DataFrame: {missing}")

    return df[list(_FINAL_COLUMNS)].drop_duplicates()


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline(input_dir: Path, output_path: Path) -> None:
    """
    Ejecuta el pipeline completo de construcción del dataset.

    Parameters
    ----------
    input_dir:
        Directorio con los TSVs de anotaciones de PharmGKB.
    output_path:
        Ruta del fichero TSV de salida.
    """
    files = [input_dir / fname for fname in _ANNOTATION_FILES]
    existing_files = [f for f in files if f.exists()]

    if not existing_files:
        ConsoleIO.print_error(f"No se encontró ningún archivo en: {input_dir}")
        logger.error("Archivos esperados: %s", [f.name for f in files])
        return

    missing_files = set(files) - set(existing_files)
    if missing_files:
        logger.warning(
            "Archivos no encontrados (se omiten): %s",
            [f.name for f in missing_files],
        )

    # 1. Cargar y filtrar
    dataframes: list[pd.DataFrame] = []
    for filepath in existing_files:
        df = load_and_filter(filepath)
        if df.empty:
            logger.warning("Archivo vacío o sin filas válidas tras filtrado: %s", filepath.name)
            continue
        df["Source_File"] = filepath.name
        dataframes.append(df)

    if not dataframes:
        ConsoleIO.print_error("Ningún archivo produjo datos válidos tras el filtrado.")
        return

    # 2. Concatenar
    # pandas ≥ 2.0: copy_on_write está disponible; ≥ 3.0: es el comportamiento por defecto
    df_master = pd.concat(dataframes, ignore_index=True)
    logger.info("Concatenación completada. Total filas iniciales: %d", len(df_master))

    # 3. Transformaciones encadenadas
    df_master = normalize_drugs(df_master)
    df_master = process_alleles(df_master)
    df_master = rename_and_fill_columns(df_master)
    df_master = build_genalle(df_master)

    # 4. Selección final y deduplicación
    try:
        df_final = select_and_deduplicate(df_master)
    except ValueError as exc:
        ConsoleIO.print_error(str(exc))
        logger.exception("Error en la selección de columnas finales.")
        return

    # 5. Exportar
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_path, sep="\t", index=False)

    ConsoleIO.print_info(f"Dataset guardado en: {output_path}")
    ConsoleIO.print_info(
        f"Dimensiones finales (Tabular Categórico para DeepFM): {df_final.shape}"
    )
    logger.info("Columnas generadas: %s", list(df_final.columns))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construye el dataset tabular PharmGKB para DeepFM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directorio con los TSV de anotaciones de PharmGKB.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Ruta de salida para el TSV procesado.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(input_dir=args.input_dir, output_path=args.output)


if __name__ == "__main__":
    main()
