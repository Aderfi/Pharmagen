import csv
import re

input_file = 'drug_names.txt'
output_file = 'drugs_filt.txt'

def clean_name(name):
    # Lowercase and strip
    name = name.lower().strip()
    # Replace underscores with spaces (as seen in some entries like Sodium_ascorbate)
    name = name.replace('_', ' ')
    
    # Handle commas
    if ',' in name:
        # Case 1: Chemical numbering like 1,7-dimethyl... -> 1-7-dimethyl...
        # Look for Digit,Digit
        name = re.sub(r'(\d),(\d)', r'\1-\2', name)
        
        # Case 2: "Name, Modifier" -> "Modifier Name"
        # Only do this if there is still a comma (meaning it wasn't just a chemical comma)
        if ',' in name:
            parts = name.split(',')
            if len(parts) == 2:
                # "ace inhibitors, plain" -> "plain ace inhibitors"
                p1 = parts[0].strip()
                p2 = parts[1].strip()
                name = f"{p2} {p1}"
            else:
                # If there are multiple commas in a single field, it's complicated.
                # e.g. "drug A, drug B, drug C" inside a single quoted field?
                # If that happens, we might just want to remove commas or replace with spaces.
                # Instruction: "reformulalos para que no tengan coma"
                # We will replace remaining commas with spaces to be safe.
                name = name.replace(',', '')
    
    # Collapse multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def main():
    processed_names = set()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Pre-process lines to handle chemical names with commas (e.g., 3,4-dimethyl...)
            # We replace digit,digit with digit-digit BEFORE csv parsing.
            lines = f.readlines()
            
        processed_lines = []
        for line in lines:
            # Replace 1,2 with 1-2
            line = re.sub(r'(\d),(\d)', r'\1-\2', line)
            processed_lines.append(line)
            
        # Use csv reader on the processed lines
        reader = csv.reader(processed_lines, skipinitialspace=True)
        for row in reader:
            for item in row:
                if item.strip():
                    cleaned = clean_name(item)
                    if cleaned:
                        processed_names.add(cleaned)
                            
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Sort and write
    sorted_names = sorted(list(processed_names))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for name in sorted_names:
            f.write(name + '\n')
            
    print(f"Processed {len(sorted_names)} unique drug names.")

if __name__ == "__main__":
    main()
