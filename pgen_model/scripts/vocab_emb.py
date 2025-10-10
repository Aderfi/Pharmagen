import json
from pathlib import Path
import os, sys

Pharmagen = Path(__file__).resolve().parents[2]  
           # Add project root to sys.path
sys.path.append(str(Pharmagen))
os.chdir(str(Pharmagen))


import src.scripts.config as cfg

def extract_vocab(json_file):
    json_file_name = str(json_file).split('/')[-1]
    json_file_base = json_file_name.split('.')[0]
    vocab = set()
    with open(json_file, encoding='utf-8') as f:
        data = json.load(f)
        try:
            for key, value in data.items():
                vocab.add(key)
                if isinstance(value, str):
                    vocab.add(value)
                elif isinstance(value, list):
                    for v in value:
                        vocab.add(v)
        except:
            for i in data:
                if isinstance(i, str):
                    vocab.add(i)

    with open(Path((cfg.MODEL_LABEL_VOCABS_DIR) / f"{json_file_base}_vocab.json"), 'w', encoding='utf-8') as f:
        json.dump(list(vocab), f, ensure_ascii=False, indent=2)
    return vocab

json_dir = cfg.PGEN_MODEL_DIR / 'docs_data' / 'json'
json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]


for i in range(len(json_files)):
    vocab = extract_vocab(json_dir / json_files[i])

vocab_files = [f for f in os.listdir(cfg.MODEL_LABEL_VOCABS_DIR) if f.endswith('_vocab.json')]
vocab1 = set()

for vf in vocab_files:
    with open(Path(cfg.MODEL_LABEL_VOCABS_DIR) / vf, 'r', encoding='utf-8') as f:
        vocab1 = vocab1.union(set(json.load(f)))

full_vocab = set().union(*[extract_vocab(json_dir / json_files[i]) for i in range(len(json_files))])
print(f"Total unique tokens: {len(full_vocab)}")



'''
#### Debug checks
print((list(vocab1)[:10]))
print((list(vocab2)[:10]))
print((list(full_vocab)[:10]))
#### End debug checks
'''
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Puedes cambiar el modelo si deseas (ej: "dmis-lab/biobert-base-cased-v1.1")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
model.eval()  # Ponemos el modelo en modo evaluación

def get_embedding(token):
    with torch.no_grad():
        inputs = tokenizer(token, return_tensors="pt", truncation=True, max_length=32, verbose=True)
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

embeddings_dir = cfg.MODEL_LABEL_VOCABS_DIR

with open(f"{embeddings_dir}/dict_emb_vocabs.pkl", 'wb') as f:
    pickle.dump(embeddings_dict, f)
