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
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.15,
    ):
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
        # Validación de parámetros
        if gamma < 0:
            raise ValueError(f"gamma debe ser >= 0, recibido: {gamma}")
        
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(f"label_smoothing debe estar en [0, 1), recibido: {label_smoothing}")
        
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"reduction debe ser 'none', 'mean' o 'sum', recibido: {reduction}")
    
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
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )
        
        # 2. Calcular probabilidades de la clase correcta (pt)
        # Aplicar log_softmax y luego exp para obtener probabilidades
        log_probs = F.log_softmax(inputs, dim=-1)
        probs = torch.exp(log_probs)
        
        # Obtener la probabilidad de la clase correcta para cada muestra
        # pt = probs[batch_idx, target_class]
        pt = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
        
        # 3. Calcular el término de enfoque: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma
        
        # 4. Aplicar Focal Loss: FL = focal_weight * CE
        focal_loss = focal_weight * ce_loss
        
        # 5. Reducción
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
    
    def extra_repr(self) -> str:
        """Representación string para debugging."""
        return (
            f"alpha={'custom' if self.alpha is not None else None}, "
            f"gamma={self.gamma}, "
            f"reduction={self.reduction}, "
            f"label_smoothing={self.label_smoothing}"
        )


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

def create_focal_loss(
    class_weights: torch.Tensor | None = None,
    gamma: float = 2.0,
    label_smoothing: float = 0.1,
    adaptive: bool = False,
) -> nn.Module:
    """
    Factory function para crear Focal Loss.
    
    Args:
        class_weights: Tensor con pesos por clase (opcional)
        gamma: Parámetro de enfoque (default: 2.0)
        label_smoothing: Label smoothing (default: 0.1)
        adaptive: Usar gamma adaptativo (default: False)
    
    Returns:
        FocalLoss o AdaptiveFocalLoss instance
    
    Ejemplo:
        >>> weights = torch.tensor([1.0, 2.5, 1.8, ...])
        >>> criterion = create_focal_loss(class_weights=weights, gamma=2.0)
    """
    if adaptive:
        return AdaptiveFocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )
    else:
        return FocalLoss(
            alpha=class_weights,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )