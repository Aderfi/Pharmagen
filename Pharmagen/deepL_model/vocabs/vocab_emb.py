import json

def extract_vocab(json_file):
    vocab = set()
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        for key, value in data.items():
            vocab.add(key)
            if isinstance(value, str):
                vocab.add(value)
            elif isinstance(value, list):
                for v in value:
                    vocab.add(v)
    return vocab

vocab1 = extract_vocab('ATC_drug_dict_ENG.json')
vocab2 = extract_vocab('drug_gene_output.json')
full_vocab = vocab1.union(vocab2)
print(f"Total unique tokens: {len(full_vocab)}")


from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Puedes cambiar el modelo si deseas (ej: "dmis-lab/biobert-base-cased-v1.1")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model.eval()  # Ponemos el modelo en modo evaluación

def get_embedding(token):
    with torch.no_grad():
        inputs = tokenizer(token, return_tensors="pt", truncation=True, max_length=32,)
        outputs = model(**inputs)
        # Usamos el embedding de la [CLS] token (primera posición)
        emb = outputs.last_hidden_state[:,0,:].squeeze().numpy()
    return emb

# Diccionario de embeddings
embeddings_dict = {}
for token in full_vocab:
    try:
        emb = get_embedding(token)
        embeddings_dict[token] = emb
    except Exception as e:
        print(f"Error procesando token {token}: {e}")

import pickle

with open('embeddings_dict.pkl', 'wb') as f:
    pickle.dump(embeddings_dict, f)