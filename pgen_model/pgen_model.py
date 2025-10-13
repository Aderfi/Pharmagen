import sys, os, json
import pandas as pd
import numpy as np
import torch
import scripts.model_structure as ms


model = PharmacoModel(n_drugs, n_genotypes, n_effects, n_outcomes, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)