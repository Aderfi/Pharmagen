from collections import Counter, defaultdict
import re
import unicodedata

import pandas as pd

from src.data.data_handler import (
    clean_dataset_content,
    load_and_clean_dataset,
)

_SUSPICIOUS_CHARS = re.compile(r"[^\w\s\-().áéíóúüñÁÉÍÓÚÜÑ]", re.UNICODE)

# Etiquetas que indican un valor vacío o desconocido y no deben tratarse como labels reales
_NULL_LABELS = frozenset({"__unknown__", "nan", "none", "", "null"})


def inspect_multilabel_parsing(
    df_raw: "pd.DataFrame",
    df_clean: "pd.DataFrame",
    col: str,
    n_samples: int = 40,
    show_all_labels: bool = True,
) -> None:
    """
    Imprime una auditoría visual del resultado del parsing regex sobre una
    columna multi-label, comparando el valor crudo con el resultado limpio.

    Para cada muestra única del dataset muestra:
      - El valor RAW original (tal como viene del CSV)
      - El valor CLEAN tras clean_dataset_content (delimitadores → '|')
      - Cada etiqueta individual tras el split
      - Un flag si la etiqueta contiene caracteres sospechosos o está vacía

    Al final imprime el catálogo completo de etiquetas únicas con su frecuencia
    y marca las problemáticas para facilitar la corrección.

    Args:
        df_raw:          DataFrame antes de clean_dataset_content (headers ya lowercase).
        df_clean:        DataFrame después de clean_dataset_content.
        col:             Nombre de la columna a inspeccionar.
        n_samples:       Número máximo de valores únicos a mostrar en detalle.
        show_all_labels: Si True muestra el catálogo completo de etiquetas únicas.
    """

    W = 72

    def _hdr(text: str):
        print(f"\n  ╔{'═' * (W - 2)}╗")
        print(f"  ║  {text:<{W - 4}}║")
        print(f"  ╚{'═' * (W - 2)}╝")

    def _sec(text: str):
        print(f"\n  ┌─ {text} {'─' * max(0, W - len(text) - 5)}┐")

    def _end_sec():
        print(f"  └{'─' * W}┘")

    def _flag_label(label: str) -> tuple[str, str]:
        """
        Devuelve (flag_emoji, motivo) para una etiqueta ya splitteada.
        Retorna ('', '') si la etiqueta es correcta.
        """
        if label.lower() in _NULL_LABELS:
            return "🔘", "valor nulo/desconocido"
        if _SUSPICIOUS_CHARS.search(label):
            bad_chars = set(_SUSPICIOUS_CHARS.findall(label))
            reprs = {repr(c) for c in bad_chars}
            return "⚠️ ", f"caracteres sospechosos: {reprs}"
        if label != label.strip():
            return "⚠️ ", "espacios al inicio o final"
        if len(label) == 1 and not label.isalnum():
            return "⚠️ ", "etiqueta de un solo carácter no alfanumérico"
        if len(label) > 120:
            return "⚠️ ", f"etiqueta muy larga ({len(label)} chars) — posible fallo de split"
        # Detectar caracteres de control (invisible corruption)
        for ch in label:
            cat = unicodedata.category(ch)
            if cat.startswith("C"):
                return "⚠️ ", f"carácter de control U+{ord(ch):04X} ({unicodedata.name(ch, '?')})"
        return "", ""

    # ------------------------------------------------------------------
    _hdr(f"INSPECCIÓN DE PARSING MULTI-LABEL  ·  columna: '{col}'")

    if col not in df_raw.columns:
        print(f"\n  ❌  Columna '{col}' no encontrada en df_raw.")
        return
    if col not in df_clean.columns:
        print(f"\n  ❌  Columna '{col}' no encontrada en df_clean.")
        return

    raw_series   = df_raw[col].astype(str)
    clean_series = df_clean[col].astype(str)

    # Estadísticas globales antes de entrar al detalle
    n_total = len(raw_series)
    n_changed = (raw_series != clean_series).sum()
    print(f"\n  Filas totales:           {n_total}")
    print(f"  Filas modificadas por regex: {n_changed}  ({n_changed/n_total*100:.1f}%)")

    # ------------------------------------------------------------------
    # Vista DETALLADA: valores únicos del dataset (limitado a n_samples)
    # ------------------------------------------------------------------
    _sec(f"DETALLE — primeros {n_samples} valores únicos (raw → clean → etiquetas)")

    # Emparejar raw con clean usando el índice compartido
    paired = pd.DataFrame({"raw": raw_series, "clean": clean_series})
    unique_pairs = paired.drop_duplicates(subset="raw").head(n_samples)

    all_labels_seen: list[str] = []
    label_issues: defaultdict[str, list[str]] = defaultdict(list)

    for _, pair_row in unique_pairs.iterrows():
        raw_val   = pair_row["raw"]
        clean_val = pair_row["clean"]
        changed   = "→" if raw_val != clean_val else " "

        print(f"\n  {'─' * W}")
        # Truncar valores largos para no romper el formato de consola
        raw_disp   = raw_val[:80]   + ("…" if len(raw_val)   > 80 else "")
        clean_disp = clean_val[:80] + ("…" if len(clean_val) > 80 else "")
        print(f"  RAW  : {raw_disp}")
        print(f"  CLEAN: {clean_disp}  {changed if raw_val != clean_val else ''}")

        # Split y auditoría de cada etiqueta
        labels = [lbl.strip() for lbl in clean_val.split("|")]
        for lbl in labels:
            flag, reason = _flag_label(lbl)
            suffix = f"  ← {reason}" if reason else ""
            marker = flag if flag else "  ✓"
            print(f"    {marker}  [{repr(lbl)}]{suffix}")
            all_labels_seen.append(lbl)
            if flag:
                label_issues[lbl].append(reason)

    _end_sec()

    if len(unique_pairs) < paired["raw"].nunique():
        remaining = paired["raw"].nunique() - len(unique_pairs)
        print(f"\n  ℹ️  {remaining} valores únicos adicionales no mostrados (usa n_samples para ampliar).")

    # ------------------------------------------------------------------
    # CATÁLOGO COMPLETO de etiquetas únicas
    # ------------------------------------------------------------------
    if show_all_labels:
        # Parsear TODAS las filas para el catálogo (no solo las n_samples)
        all_labels_full: list[str] = []
        for val in clean_series:
            for lbl in val.split("|"):
                all_labels_full.append(lbl.strip())

        label_counts = Counter(all_labels_full)
        n_unique = len(label_counts)

        _sec(f"CATÁLOGO COMPLETO — {n_unique} etiquetas únicas")
        print(f"\n  {'#':>4}  {'Etiqueta':<45} {'Count':>7}  {'%':>6}  Estado")
        print(f"  {'─'*4}  {'─'*45} {'─'*7}  {'─'*6}  {'─'*20}")

        total_instances = sum(label_counts.values())
        for rank, (lbl, count) in enumerate(label_counts.most_common(), start=1):
            pct = count / total_instances * 100
            flag, reason = _flag_label(lbl)
            estado = f"{flag} {reason}" if flag else "✅ ok"
            lbl_disp = lbl[:44] + ("…" if len(lbl) > 44 else "")
            print(f"  {rank:>4}  {lbl_disp:<45} {count:>7}  {pct:>5.1f}%  {estado}")

        _end_sec()

    # ------------------------------------------------------------------
    # RESUMEN DE PROBLEMAS ENCONTRADOS
    # ------------------------------------------------------------------
    _sec("RESUMEN DE PROBLEMAS")

    if not label_issues:
        print("\n  ✅  Ningún problema detectado en los valores muestreados.")
    else:
        print(f"\n  ⚠️  {len(label_issues)} etiqueta(s) con problemas:\n")
        for lbl, reasons in label_issues.items():
            unique_reasons = list(dict.fromkeys(reasons))  # deduplicar preservando orden
            print(f"    [{repr(lbl)}]")
            for r in unique_reasons:
                print(f"      → {r}")

    # Advertencias globales adicionales
    if show_all_labels:
        null_count = sum(c for lbl, c in label_counts.items() if lbl.lower() in _NULL_LABELS)
        if null_count:
            pct_null = null_count / total_instances * 100
            print(f"\n  ⚠️  {null_count} instancias de etiquetas nulas/desconocidas ({pct_null:.1f}% del total).")

        single_char = [(lbl, c) for lbl, c in label_counts.items()
                       if len(lbl) == 1 and not lbl.isalnum()]
        if single_char:
            print(f"\n  ⚠️  {len(single_char)} etiqueta(s) de un solo carácter no alfanumérico:")
            for lbl, c in single_char:
                print(f"      [{repr(lbl)}]  →  {c} instancias")

    _end_sec()
    print()



df_raw = pd.read_csv(
    "data/train/final_test_genalle.tsv",
    sep="\t",
    dtype=str)

df_raw.columns = df_raw.columns.str.lower().str.strip()
df_clean = clean_dataset_content(df_raw.copy(), multi_label_cols=["phenotype_outcome"])

inspect_multilabel_parsing(df_raw, df_clean, col="phenotype_outcome", n_samples=50)
