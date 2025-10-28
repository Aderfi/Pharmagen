import csv
import glob
import os
import sys

csv.field_size_limit(2**31 - 1)


def convert_tsv_to_csv(tsv_file, csv_file):
    with open(tsv_file, "r", newline="", encoding="utf-8") as tsv_in, open(
        csv_file, "w", newline="", encoding="utf-8"
    ) as csv_out:
        reader = csv.reader(tsv_in, delimiter="\t")
        writer = csv.writer(csv_out)
        for row in reader:
            writer.writerow(row)


def main():
    tsv_files = glob.glob("*.tsv")
    for tsv_file in tsv_files:
        csv_file = os.path.splitext(tsv_file)[0] + ".csv"
        convert_tsv_to_csv(tsv_file, csv_file)
        print(f"Converted {tsv_file} to {csv_file}")


if __name__ == "__main__":
    main()
