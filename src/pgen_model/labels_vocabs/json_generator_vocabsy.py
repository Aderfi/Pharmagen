import pandas as pd
import torch
import json
from transformers import AutoTokenizer, AutoModel


# Cargar el CSV
df = pd.read_csv("var_drug_ann_with_ATC.csv", sep=";")

# Define los pares que quieres embeddizar
outcome_variation_pairs = list(
    zip(df["Outcome"].astype(str), df["Variation"].astype(str))
)
variation_variation1_pairs = list(
    zip(df["Variation"].astype(str), df["Variation_1"].astype(str))
)
variation1_effect_pairs = list(
    zip(df["Variation_1"].astype(str), df["Outcome"].astype(str))
)  # Cambia 'Outcome' si tienes otra columna

# Drug_gene from csv

with open("drug_gene.json", "r") as f:
    drug_gene_dict = json.load(f)


# Concatenar usando [SEP]
def pair_to_str(pair):
    return f"{pair[0]} [SEP] {pair[1]}"


outcome_variation_strs = [pair_to_str(pair) for pair in outcome_variation_pairs]
variation_variation1_strs = [pair_to_str(pair) for pair in variation_variation1_pairs]
variation1_effect_strs = [pair_to_str(pair) for pair in variation1_effect_pairs]

# Cargar BioBERT
model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
model = model.to(device)


# Función para obtener el embedding [CLS] de cada par
def get_biobert_embeddings(sentences, tokenizer, model, device="cuda", batch_size=32):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
    return embeddings


# Obtener los embeddings
outcome_variation_embeddings = get_biobert_embeddings(
    outcome_variation_strs, tokenizer, model, device
)
variation_variation1_embeddings = get_biobert_embeddings(
    variation_variation1_strs, tokenizer, model, device
)
variation1_effect_embeddings = get_biobert_embeddings(
    variation1_effect_strs, tokenizer, model, device
)


def get_genes(drug):
    genes = drug_gene_dict.get(str(drug), [])
    if not genes:
        return "NOGENE"
    return ",".join(genes)


drug_gene_strs = [
    f"{row['Drug']} [SEP] {get_genes(row['Drug'])}" for _, row in df.iterrows()
]
drug_gene_embeddings = get_biobert_embeddings(drug_gene_strs, tokenizer, model, device)

# Ejemplo: imprime el embedding del primer par
print("Embedding de Outcome-Variation, primer par:", outcome_variation_embeddings[0])

# Guarda los embeddings si lo deseas
import numpy as np

np.save("outcome_variation_embeddings_biobert.npy", outcome_variation_embeddings)
np.save("variation_variation1_embeddings_biobert.npy", variation_variation1_embeddings)
np.save("variation1_effect_embeddings_biobert.npy", variation1_effect_embeddings)
np.save("drug_gene_embeddings_biobert.npy", drug_gene_embeddings)
