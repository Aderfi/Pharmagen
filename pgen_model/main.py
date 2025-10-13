import scripts
from scripts import *
import torch
import torch.nn as nn
from pgen_model import *

class Predictions:
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def evaluate
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                drug = batch['drug'].to(self.device)
                genotype = batch['genotype'].to(self.device)
                effect = batch['effect'].to(self.device)
                entity = batch['entity'].to(self.device)

                effect_pred, entity_pred = self.model(drug, genotype)
                # Compute metrics here