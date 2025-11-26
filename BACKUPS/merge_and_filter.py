import csv
import sys

# Increase CSV field limit
max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 2)

input_files = ['var_drug_ann_long.csv', 'var_fa_ann_long.csv', 'var_pheno_ann_long.csv']
output_file = 'data_fil.csv'

# Target columns to keep in the final file
# We will use a superset, filling missing values with empty string
target_columns = ['ATC', 'Drug', 'Gene', 'Genotype', 'Alleles', 'Effect', 'Outcome', 'Sentence', 'Population']

def is_significant(row):
    # In the absence of explicit "Significance" or "Associated" columns,
    # we infer positive association from the Sentence text.
    # Most entries say "is associated with".
    # We exclude explicit negatives.
    
    sentence = row.get('Sentence', '').lower()
    notes = row.get('Notes', '').lower()
    
    # Explicit columns check (if they existed, but they don't seem to in the header)
    # Just in case the DictReader finds them
    sig = row.get('Significance', '').lower()
    assoc = row.get('Associated', '').lower()
    
    if sig == 'yes':
        return True
    if assoc == 'positive':
        return True
    if sig == 'no':
        return False
    
    # Inference
    if 'associated with' in sentence:
        if 'not associated' in sentence or 'no association' in sentence:
             return False
        # "no significant difference" is sometimes used to describe a lack of association
        if 'no significant difference' in sentence:
            return False
        return True
    
    return False

rows_written = 0
try:
    with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=target_columns, delimiter=';')
        writer.writeheader()
        
        for file_path in input_files:
            print(f"Processing {file_path}...")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as infile:
                    reader = csv.DictReader(infile, delimiter=';')
                    
                    for row in reader:
                        if is_significant(row):
                            out_row = {col: row.get(col, '') for col in target_columns}
                            writer.writerow(out_row)
                            rows_written += 1
                            
            except FileNotFoundError:
                print(f"File not found: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    print(f"Processing complete. {rows_written} rows written to {output_file}.")

except Exception as e:
    print(f"Critical error: {e}")
