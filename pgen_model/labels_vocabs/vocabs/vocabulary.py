import json

with open("ATC_drug_dict_ENG.json") as f:
    drug_gene = json.load(f)

pairs = []
for drug, genes in drug_gene.items():
    pairs.append((drug, genes))

# Guardar a archivo vocab_pares.txt
with open("vocab_ATC_drug_pares.txt", "w", encoding="utf-8") as f:
    for drug, gene in pairs:
        f.write(f"{drug},{gene}\n")
