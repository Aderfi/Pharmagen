"""
losses.py

Implementation of Advanced Loss Functions for Imbalanced and Multi-Task Learning.
Includes Focal Loss, Asymmetric Loss, PolyLoss, and Kendall&Gal Uncertainty Weighting.
"""

import torch
import torch.nn.functional as F
from torch import nn

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Down-weights well-classified examples to focus training on hard negatives.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma (float): Focusing parameter. Higher values reduce the loss for easy examples.
        label_smoothing (float): Label smoothing factor [0.0, 1.0].
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class AdaptiveFocalLoss(FocalLoss):
    """
    Dynamically adjusts Gamma based on batch-wise accuracy.
    
    If accuracy is high, gamma decreases (standard CE behavior).
    If accuracy is low, gamma increases (hard mining behavior).
    
    Note: strictly for Single-Label Multi-Class tasks.
    """
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            acc = (logits.argmax(1) == targets).float().mean()
            # Mapping: Acc 1.0 -> Gamma 1.0 | Acc 0.0 -> Gamma 3.0
            self.gamma = float(3.0 - (acc * 2.0))
        return super().forward(logits, targets)

class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss (ASL) for Multi-Label Classification.
    
    Decouples the focusing mechanisms for positive and negative examples.
    Reference: Ben-Baruch et al. "Asymmetric Loss for Multi-Label Classification" (ICCV 2021).
    
    Args:
        gamma_neg (float): Focusing parameter for negative examples (usually high, e.g. 4.0).
        gamma_pos (float): Focusing parameter for positive examples (usually low, e.g. 1.0).
        clip (float): Probability margin to completely discard easy negatives (Hard Thresholding).
    """
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Targets must be float for multi-label BCE context
        targets = targets.float()
        
        # Sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Probabilities for Positive and Negative cases
        xs_pos = probs
        xs_neg = 1 - probs

        # Asymmetric Clipping (Hard Attention on Negatives)
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
    PolyLoss: Polynomial Expansion of Cross Entropy.
    
    Framework: Poly-1 (L_CE + epsilon * (1-Pt))
    Reference: Leng et al. (ICLR 2022).
    
    Args:
        epsilon (float): Coefficient for the polynomial term.
    """
    def __init__(self, epsilon: float = 1.0, reduction: str = 'mean', label_smoothing: float = 0.0):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, reduction='none', label_smoothing=self.smoothing
        )
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
    Homoscedastic Uncertainty Weighting for Multi-Task Learning.
    
    Learns the relative weights of multiple loss functions automatically.
    Reference: Kendall & Gal (CVPR 2018).
    
    Formula: Loss = Sum( Loss_i * exp(-sigma_i) + sigma_i )
    """
    def __init__(self, tasks: list[str]):
        super().__init__()
        # Initialize log variance (sigma) to 0
        self.log_sigmas = nn.ParameterDict({
            t: nn.Parameter(torch.zeros(1)) for t in tasks
        })

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        for task, loss in losses.items():
            sigma = self.log_sigmas[task]
            total_loss += loss * torch.exp(-sigma) + sigma
        return total_loss
