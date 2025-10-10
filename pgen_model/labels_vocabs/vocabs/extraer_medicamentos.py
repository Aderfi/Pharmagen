import json

# Carga el archivo JSON
with open('ATC_drug_dict_ENG.json', 'r', encoding='utf-8') as f:
    drug_gene_dict = json.load(f)

# Extrae las claves (drugs) y ordénalas si quieres
drugs = sorted(drug_gene_dict.keys())

# Escribe el vocabulario a un archivo .txt, uno por línea
with open('ATC_vocab.txt', 'w', encoding='utf-8') as f:
    for drug in drugs:
        f.write(f"{drug}\n")

print("Vocabulario de fármacos guardado en drug_vocab.txt")