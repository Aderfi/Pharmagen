import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import json
import optuna
from optuna import Trial as trial
from optuna import Study as study
from class_Model import PGenDataset, PGenModel

OPTUNA_RESULTS = ('optuna_outputs/results.txt')

def objective(trial, df):
    # Define the search space
    emb_dim = trial.suggest_int("emb_dim", 32, 256)
    hidden_dim = trial.suggest_int("hidden_dim", 64, 512)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    pgen_dataset = PGenDataset(df)

    # Create the model
    model = PGenModel(n_drugs, n_genotypes, n_outcomes, n_variations, n_effects, n_entitys,
                      emb_dim, hidden_dim, dropout_rate, device=device)

    # Train the model
    model.train(epochs=10, lr=1e-3)

    # Evaluate the model
    val_loss = model.evaluate(val_loader)

    return val_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
print("Mejores hiperparámetros:", study.best_params)

with open("optuna_results.txt", "a") as f:
    f.write("\n---- Trial Modelo Outcome-Variation-Effect-Entity-----\n")
    f.write(f"Mejores hiperparámetros: {study.best_params}\n")