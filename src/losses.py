# Pharmagen - Loss Functions
# Collection of custom loss functions for PGenModel.
# Includes specific logic for Multi-Label (Asymmetric) and Imbalanced (Focal/Poly) tasks.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class FocalLoss(nn.Module):
    """Optimized Focal Loss for imbalance."""
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class AdaptiveFocalLoss(FocalLoss):
    """Dynamically adjusts gamma based on batch accuracy."""
    def forward(self, logits, targets):
        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean()
            # High acc -> Low gamma (focus less). Low acc -> High gamma.
            self.gamma = float(3.0 - (acc * 2.0)) # Range [1.0, 3.0]
        return super().forward(logits, targets)

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification.
    Reduces the weight of easy negatives.
    Reference: Ben-Baruch et al. "Asymmetric Loss for Multi-Label Classification"
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=1.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        # Targets must be float for multi-label
        targets = targets.float()
        
        # Sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Positive and Negative probabilities
        xs_pos = probs
        xs_neg = 1 - probs

        # Asymmetric Clipping for negatives
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Positive term: -y * log(p) * (1-p)^gamma_pos
        p_pos = xs_pos.clamp(min=self.eps)
        loss_pos = -targets * torch.log(p_pos) * ((1 - xs_pos) ** self.gamma_pos)
        
        # Negative term: -(1-y) * log(1-p_clipped) * (p_clipped)^gamma_neg
        p_neg = xs_neg.clamp(min=self.eps)
        loss_neg = -(1 - targets) * torch.log(p_neg) * ((1 - xs_neg) ** self.gamma_neg)
        
        loss = loss_pos + loss_neg
        return loss.mean()

class PolyLoss(nn.Module):
    """
    PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions.
    Framework: Poly-1 (L_CE + epsilon * (1-Pt))
    Reference: Leng et al. (ICLR 2022)
    """
    def __init__(self, epsilon=1.0, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.smoothing = label_smoothing

    def forward(self, logits, targets):
        # CrossEntropy serves as the base (0-th order expansion)
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', label_smoothing=self.smoothing
        )
        # Pt is the probability of the ground truth class
        pt = torch.exp(-ce_loss)
        
        # Poly-1 formulation
        poly_loss = ce_loss + self.epsilon * (1 - pt)
        
        if self.reduction == 'mean':
            return poly_loss.mean()
        elif self.reduction == 'sum':
            return poly_loss.sum()
        return poly_loss

class MultiTaskUncertaintyLoss(nn.Module):
    """
    Kendall & Gal (2018): Learns weights for multi-task loss.
    Loss = Loss_i * exp(-sigma_i) + sigma_i
    """
    def __init__(self, tasks: List[str]):
        super().__init__()
        self.log_sigmas = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in tasks
        })

    def forward(self, losses: Dict[str, torch.Tensor]):
        total_loss = 0.0
        for task, loss in losses.items():
            sigma = self.log_sigmas[task]
            total_loss += loss * torch.exp(-sigma) + sigma
        return total_loss
