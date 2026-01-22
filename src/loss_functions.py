# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss para Multi-Class Classification
    ===========================================
    Implementación optimizada para PyTorch con soporte para class weights.

    Referencia: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    Principalmente orientado a casos multi-target con desbalanceo de clases.
    """
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        # Registramos alpha como buffer para que se mueva con el modelo (cuda/cpu)
        # pero no sea un parámetro entrenable.
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Mover alpha al dispositivo correcto si es necesario (manejado por register_buffer usualmente)
        weight = cast(torch.Tensor | None, self.alpha)
        
        # Cross Entropy con Label Smoothing
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class AdaptiveFocalLoss(FocalLoss):
    """
    Focal Loss que ajusta 'gamma' dinámicamente según la accuracy del batch actual.
    Útil cuando el entrenamiento es inestable al principio.
    """
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        gamma_min: float = 1.0,
        gamma_max: float = 3.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__(alpha, gamma, reduction, label_smoothing)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cálculo rápido de accuracy sin romper el grafo de computación
        with torch.no_grad():
            predictions = torch.argmax(inputs, dim=1)
            # Accuracy escalar
            batch_acc = (predictions == targets).float().mean()
            
            # Mapeo lineal inverso: +Accuracy -> -Gamma (batch fácil, enfocar menos)
            # Clamp para asegurar límites
            new_gamma = self.gamma_max - (batch_acc * (self.gamma_max - self.gamma_min))
            self.gamma = float(torch.clamp(new_gamma, self.gamma_min, self.gamma_max))

        return super().forward(inputs, targets)


class GeometricLoss(nn.Module): # Opuesto de Uncertainty Loss
    """
    Estrategia de Pérdida Geométrica para Multi-Task Learning.
    Penaliza tareas con pérdida muy alta más severamente que la suma aritmética.

    Principalmente orientado a casos multi-target con desbalanceo severo.
    """
    def __init__(self):
        super().__init__()

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        if not losses:
            return torch.tensor(0.0, requires_grad=True)

        # Obtener dispositivo del primer tensor
        first_loss = next(iter(losses.values()))
        device = first_loss.device
        
        log_sum = torch.tensor(0.0, device=device)
        n_tasks = len(losses)

        for loss in losses.values():
            # Estabilidad numérica: clamp para evitar log(0)
            safe_loss = torch.clamp(loss, min=1e-7)
            log_sum = log_sum + torch.log(safe_loss)

        # Exp(Mean(Log(Losses)))
        return torch.exp(log_sum / n_tasks)


class MultiTaskUncertaintyLoss(nn.Module):
    """
    Implementación de Kendall & Gal (2018) para ponderación automática de pérdidas.
    
    Mantiene parámetros entrenables (log_sigma) para cada tarea.
    Aprende a 'confiar menos' (sigma alto) en tareas difíciles o ruidosas.
    """
    def __init__(self, task_names: list[str], priorities: dict[str, float] | None = None):
        super().__init__()
        self.task_names = task_names
        self.priorities = priorities or {}
        
        # Creamos un ParameterDict para que el optimizador vea estos pesos
        self.log_sigmas = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def forward(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        for task, loss in losses.items():
            if task not in self.log_sigmas:
                # Fallback si llega una tarea no registrada (seguridad)
                total_loss = total_loss + loss
                continue

            log_sigma = self.log_sigmas[task]
            priority = self.priorities.get(task, 1.0)

            # Fórmula: Loss * exp(-log_sigma) + log_sigma
            # El término log_sigma actúa como regularizador para evitar que exp(-sigma) crezca infinito
            weighted_loss = (loss * priority) * torch.exp(-log_sigma) + log_sigma
            total_loss = total_loss + weighted_loss

        return total_loss

    def get_sigmas(self) -> dict[str, float]:
        """Retorna los valores actuales de incertidumbre (interpretables)."""
        return {k: float(torch.exp(v).item()) for k, v in self.log_sigmas.items()}