import pandas as pd
import json
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Configuración
CSV_FILES = [
    "var_drug_ann_with_ATC.csv",
    "var_fa_ann_with_ATC.csv",
    "var_pheno_ann_with_ATC.csv",
]
CSV_COLUMNS = [
    ["ATC","Drug","Gene","Genotype","Alleles","Outcome","Variation","Variation_1","Sentence","Notes","Population","Ref_Genotype"],
    ["ATC","Drug","Gene","Genotype","Alleles","Outcome","Variation","Variation_1","Entity","Sentence","Notes"],
    ["ATC","Drug","Gene","Genotype","Alleles","Outcome","Variation","Variation_1","Effect","RAM","Sentence","Notes","Population","Ref_Genotype"],
]
VOCAB_COLUMNS = {
    "var_drug_ann_with_ATC.csv": ["Drug","Gene","Genotype","Alleles","Outcome","Variation","Variation_1"],
    "var_fa_ann_with_ATC.csv": ["Drug","Gene","Genotype","Alleles","Outcome","Variation","Variation_1","Entity"],
    "var_pheno_ann_with_ATC.csv": ["Drug","Gene","Genotype","Alleles","Outcome","Variation","Variation_1","Effect","RAM"],
}
JSON_FILES = [
    'ATC_drug_dict_ENG.json',
    'drug_gene_output.json',
]

BIOBERT_MODEL = "dmis-lab/biobert-base-cased-v1.1"

# Usa GPU si está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(BIOBERT_MODEL)
model = AutoModel.from_pretrained(BIOBERT_MODEL).to(device)
model.eval()

def get_vocab_from_column(col_values):
    return sorted(set([str(val) for val in col_values if pd.notnull(val)]))

def save_vocab(vocab, name):
    with open(f'vocab_{name}.txt', 'w', encoding='utf-8') as f:
        for token in vocab:
            f.write(f"{token}\n")

@torch.no_grad()
def get_biobert_embedding(text):
    # Tokenización y obtención de embedding [CLS]
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Usar el embedding [CLS] (primer token)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu()
    return cls_embedding

def save_embeddings(vocab, name):
    embeddings = {}
    all_embs = []
    for token in tqdm(vocab, desc=f"Embeddings {name}"):
        emb = get_biobert_embedding(token)
        embeddings[token] = emb
        all_embs.append(emb.unsqueeze(0))
    all_embs = torch.cat(all_embs, dim=0)
    torch.save({"tokens": vocab, "embeddings": all_embs}, f"embedding_{name}.pt")

def process_csv_file(csv_file, columns, vocab_columns):
    print(f"Leyendo {csv_file} ...")
    df = pd.read_csv(csv_file, sep=";", usecols=columns)
    for col in vocab_columns:
        if col in df.columns:
            vocab = get_vocab_from_column(df[col])
            print(f"Columna: {col}, Tokens únicos: {len(vocab)}")
            save_vocab(vocab, f"{os.path.splitext(csv_file)[0]}_{col}")
            save_embeddings(vocab, f"{os.path.splitext(csv_file)[0]}_{col}")

def process_json_file(json_file):
    print(f"Leyendo {json_file} ...")
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys_vocab = sorted(set(data.keys()))
    save_vocab(keys_vocab, f"{os.path.splitext(json_file)[0]}_keys")
    save_embeddings(keys_vocab, f"{os.path.splitext(json_file)[0]}_keys")
    values_vocab = set()
    for v in data.values():
        if isinstance(v, list):
            values_vocab.update(v)
        elif isinstance(v, dict):
            values_vocab.update(v.keys())
        else:
            values_vocab.add(str(v))
    values_vocab = sorted(values_vocab)
    save_vocab(values_vocab, f"{os.path.splitext(json_file)[0]}_values")
    save_embeddings(values_vocab, f"{os.path.splitext(json_file)[0]}_values")

if __name__ == "__main__":
    for i, csv_file in enumerate(CSV_FILES):
        columns = CSV_COLUMNS[i]
        vocab_columns = VOCAB_COLUMNS[csv_file]
        process_csv_file(csv_file, columns, vocab_columns)
    for json_file in JSON_FILES:
        process_json_file(json_file)
    print("Listo. Se han generado los vocabs y embeddings de cada columna seleccionada usando BioBERT.")