from pathlib import Path
import pandas as pd
import csv
import sys
import os

from src.config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR


def increase_csv_field_limit():
    """Aumenta el límite de tamaño de campo para el lector CSV de Python."""
    max_int = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 2)


# Función principal para convertir archivos .tsv a .csv


def convert_tsv_to_csv(raw_dir, processed_csv_dir, cache_dir):
    """Convierte todos los archivos .tsv en raw_dir a .csv en processed_csv_dir."""
    raw_dir = RAW_DATA_DIR
    processed_csv_dir = PROCESSED_DATA_DIR / "csv"
    cache_dir = CACHE_DIR

    # Encuentra todos los archivos .tsv en RAW_DATA_DIR
    tsv_files = list(raw_dir.glob("*.tsv"))
    txt_temp_files = []

    for tsv_file in tsv_files:
        csv_file = processed_csv_dir / (tsv_file.stem + ".csv")
        txt_temp_file = cache_dir / (tsv_file.stem + ".txt")
        txt_temp_files.append(txt_temp_file)

        try:
            print(f"▶️ Leyendo '{tsv_file.name}'...")
            df = pd.read_csv(tsv_file, sep="\t", engine="python")
            df.to_csv(csv_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"✅ '{tsv_file.name}' convertido a '{csv_file.name}'.")

            # Escribir cabeceras numeradas en archivo temporal .txt
            columns = df.columns.tolist()
            with open(txt_temp_file, "w", encoding="utf-8") as out_f:
                for idx, col in enumerate(columns):
                    out_f.write(f"{idx}\t{col}\n")
            print(f"📝 Cabeceras guardadas en '{txt_temp_file.name}' (temporal).")

        except Exception as e:
            print(f"❌ Error procesando '{tsv_file.name}': {e}")

    return txt_temp_files


######  LIMPIEZA DE ARCHIVOS TEMPORALES  ######
def cleanup_temp_files(temp_files):
    """Elimina los archivos temporales .txt generados en cache_dir."""
    for f in temp_files:
        try:
            os.remove(f)
            print(f"🗑️ Eliminado archivo temporal: {Path(f).name}")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar '{f}': {e}")


if __name__ == "__main__":
    increase_csv_field_limit()
    # Ajusta el subdirectorio "csv" dentro de PROCESSED_DATA_DIR si no existe
    processed_csv_dir = PROCESSED_DATA_DIR / "csv"
    processed_csv_dir.mkdir(parents=True, exist_ok=True)
    # Ejecuta conversión y generación de archivos temporales
    txt_temp_files = convert_tsv_to_csv(RAW_DATA_DIR, processed_csv_dir, CACHE_DIR)

    # Limpia los archivos temporales generados

    cleanup_temp_files(txt_temp_files)
    print("\n----- Proceso finalizado correctamente. -----\n")
