"""
Focal Loss para Multi-Class Classification
===========================================
Implementación optimizada para PyTorch con soporte para class weights.

Referencia: Lin et al. "Focal Loss for Dense Object Detection" (2017)
https://arxiv.org/abs/1708.02002
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación multi-clase con desbalanceo severo.
    
    Fórmula: FL(pt) = -αt * (1 - pt)^γ * log(pt)
    
    Parámetros:
        alpha (Tensor | None): Pesos por clase (class weights).
                               Si es None, todas las clases tienen peso 1.0.
                               Shape: [num_classes]
        
        gamma (float): Parámetro de enfoque. 
                       - γ = 0: Equivalente a CrossEntropyLoss
                       - γ = 2: Configuración estándar (recomendado)
                       - γ > 2: Enfoque más agresivo en ejemplos difíciles
        
        reduction (str): Especifica la reducción a aplicar al output:
                         'none' | 'mean' | 'sum'
        
        label_smoothing (float): Label smoothing (opcional, 0.0 a 1.0)
    
    Ejemplo:
        >>> num_classes = 10
        >>> class_weights = torch.tensor([1.0, 2.0, ...])  # Pesos calculados
        >>> criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        >>> 
        >>> logits = model(inputs)  # Shape: [batch_size, num_classes]
        >>> targets = ...           # Shape: [batch_size]
        >>> loss = criterion(logits, targets)
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:        
        """
        Args:
            inputs: Logits del modelo (sin softmax/sigmoid)
                    Shape: [batch_size, num_classes]
            targets: Etiquetas de clase (enteros)
                     Shape: [batch_size]
        
        Returns:
            loss: Scalar tensor (si reduction='mean' o 'sum')
                  o tensor de shape [batch_size] (si reduction='none')
        """
        # 1. Calcular CrossEntropy con label smoothing (sin reducción)
        # Mover alpha al dispositivo correcto si existe
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        
        pt = torch.exp(-ce_loss) # Cross Entropy = -log(pt) -> pt = exp(-CE)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class AdaptiveFocalLoss(FocalLoss):
    """
    Focal Loss con gamma adaptativo basado en la dificultad del batch.
    
    Ajusta automáticamente gamma según:
    - Si el batch es fácil (alta accuracy) → gamma bajo
    - Si el batch es difícil (baja accuracy) → gamma alto
    
    Parámetros adicionales:
        gamma_min (float): Gamma mínimo (default: 1.0)
        gamma_max (float): Gamma máximo (default: 3.0)
    """
    
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        gamma_min: float = 1.0,
        gamma_max: float = 3.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
    ):
        super().__init__(alpha, gamma, reduction, label_smoothing)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.base_gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calcular accuracy del batch
        predictions = torch.argmax(inputs, dim=1)
        batch_accuracy = (predictions == targets).float().mean()
        
        # Ajustar gamma inversamente proporcional a la accuracy
        # Alta accuracy → gamma bajo (enfoque ligero)
        # Baja accuracy → gamma alto (enfoque agresivo)
        self.gamma = self.gamma_max - (batch_accuracy * (self.gamma_max - self.gamma_min))
        
        return super().forward(inputs, targets)


# ══════════════════════════════════════════════════════════════
# FUNCIÓN HELPER PARA CREAR FOCAL LOSS FÁCILMENTE
# ══════════════════════════════════════════════════════════════

def create_focal_loss(class_weights: Optional[torch.Tensor] = None, gamma: float = 2.0) -> FocalLoss:
    return FocalLoss(alpha=class_weights, gamma=gamma)