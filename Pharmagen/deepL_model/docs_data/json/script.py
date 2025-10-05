'''
import asyncio
import json
from googletrans impor

# Load your ATC_drugs dictionary from the JSON archive
with open('ATC_drugs.json', 'r', encoding='utf-8') as f:
    ATC_drugs = json.load(f)

translator = Translator()

def translate_value(value):
    if isinstance(value, str):
        # Translate a single string
        return translator.translate(value, src='es', dest='en').text
    elif isinstance(value, list):
        # Translate each item in the list
        return [translator.translate(item, src='es', dest='en').text if isinstance(item, str) else item for item in value]
    else:
        # Other types remain unchanged
        return value

# Translate all values in the dictionary
translated_ATC_drugs = {k: translate_value(v) for k, v in ATC_drugs.items()}

# Optionally, save to a new JSON file
with open('ATC_drugs_en.json', 'w', encoding='utf-8') as f:
    json.dump(translated_ATC_drugs, f, ensure_ascii=False, indent=2)

print("Translation complete. Output saved to ATC_drugs_en.json.")
''''''
import json

with open('ATC_farmaco_ESP_corregido.json', 'r', encoding='utf-8') as f:
    ATC_farmaco = json.load(f)
    ATC_farmaco = [{v:k for k,v in entry.items()} for entry in ATC_farmaco]
    
with open('list_drugs.txt', 'w', encoding='utf-8') as f:
    drugs_list = [list(entry.keys()) for entry in ATC_farmaco]
    f.write('\n'.join(str(drug[0]) for drug in drugs_list))
'''
import csv
import json

# Build translation dict from ESP_ENG_Drugs.csv
translation = {}
with open('ESP_ENG_Drugs.csv', 'r', encoding='utf-8') as infile:
    csvreader = csv.reader(infile, delimiter=';')
    for row in csvreader:
        if len(row) >= 2:
            esp, eng = row[0].strip(), row[1].strip()
            translation[esp] = eng

# Translate the JSON
with open('ATC_farmaco_ESP_corregido.json', 'r', encoding='utf-8') as json_file:
    json_drug = json.load(json_file)

atc_drug_eng_list = []

for entry in json_drug:
    for atc_code, esp_name in entry.items():
        eng_name = translation.get(esp_name, esp_name)  # Default to ESP if not found
        atc_drug_eng_list.append({atc_code: eng_name.upper()})

# Optionally, to save the translated list:
with open('ATC_farmaco_ENG_corregido.json', 'w', encoding='utf-8') as outfile:
    json.dump(atc_drug_eng_list, outfile, indent=2)