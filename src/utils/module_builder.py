# Pharmagen - Pharmacogenetic Prediction and Therapeutic Efficacy
# Copyright (C) 2025 Adrim Hamed Outmani
#src/utils/factory.py
import logging
from typing import Any, Dict, List, Optional, Set, Type, Union
import torch
import torch.nn as nn
from src.losses import (
    AdaptiveFocalLoss, FocalLoss, GeometricLoss, 
    BinaryFocalLoss, AsymmetricLoss
)

logger = logging.getLogger(__name__)

class ComponentFactory:
    """
    Base Factory class implementing a Registry pattern.
    Adheres to OCP: New components can be registered without modifying the factory logic.
    """
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, component: Any) -> None:
        cls._registry[name] = component

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry.get(name)


class OptimizerFactory(ComponentFactory):
    _registry = {
        "adamw": torch.optim.AdamW,
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }

    @staticmethod
    def create(
        model: nn.Module, 
        params: Dict[str, Any], 
        uncertainty_module: Optional[nn.Module] = None
    ) -> torch.optim.Optimizer:
        
        lr = params.get("learning_rate", 1e-3)
        wd = params.get("weight_decay", 1e-4)
        opt_name = params.get("optimizer_type", "adamw").lower()
        
        # Parameter Groups construction
        param_groups = [{'params': model.parameters(), 'weight_decay': wd, 'lr': lr}]
        
        # Uncertainty module handling (Special case handled cleanly)
        if uncertainty_module:
            param_groups.append({
                'params': uncertainty_module.parameters(), 
                'weight_decay': 0.0, 
                'lr': params.get("loss_learning_rate", lr)
            })

        optimizer_cls = OptimizerFactory.get(opt_name) or torch.optim.Adam
        
        kwargs = {}
        if opt_name == "sgd":
            kwargs["momentum"] = params.get("momentum", 0.9)
            
        return optimizer_cls(param_groups, **kwargs)


class SchedulerFactory(ComponentFactory):
    _registry = {
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    }

    @staticmethod
    def create(
        optimizer: torch.optim.Optimizer, 
        params: Dict[str, Any]
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        
        stype = params.get("scheduler_type", "plateau").lower()
        
        if stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode="min", 
                factor=params.get("scheduler_factor", 0.5),
                patience=params.get("scheduler_patience", 3),
            )
        elif stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=params.get("epochs", 50), 
                eta_min=1e-6
            )
        return None


class LossFactory(ComponentFactory):
    """
    Factory specifically for creating Loss Functions (Criteria).
    Replaces the old PGenLoss and CRITERIONS_MAP.
    """
    _registry = {
        "cross_entropy": nn.CrossEntropyLoss,
        "bce_w/logits": nn.BCEWithLogitsLoss,
        "bce": nn.BCEWithLogitsLoss,
        "adapt_focal": AdaptiveFocalLoss,
        "adaptive_focal": AdaptiveFocalLoss,
        "focal": FocalLoss,
        "binary": nn.BCELoss,
        "binary_focal": BinaryFocalLoss,
        "asymmetric": AsymmetricLoss,
        "asl": AsymmetricLoss,
        "geometric": GeometricLoss,
    }

    @staticmethod
    def create_single(
        name: str, 
        params: Dict[str, Any], 
        **kwargs
    ) -> nn.Module:
        loss_cls = LossFactory.get(name)
        if not loss_cls:
            raise ValueError(f"Loss type '{name}' not registered.")

        # Logic to extract parameters relevant to specific losses could be refined here
        # For KISS, we pass specific kwargs or extract from params
        return loss_cls(**kwargs)

    @staticmethod
    def create_task_criterions(
        target_cols: List[str], 
        multi_label_cols: Set[str], 
        params: Dict[str, Any], 
        device: torch.device,
        class_pos_weights: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, nn.Module]: 
        """
        Orchestrates the creation of multiple loss functions for multi-task setups.
        """
        criterions = {}
        default_multilabel = params.get("loss_multilabel", "asymmetric")
        default_singlelabel = params.get("loss_singlelabel", "focal")
        
        for col in target_cols:
            is_multilabel = col in multi_label_cols
            loss_key = default_multilabel if is_multilabel else default_singlelabel

            # Fallback logic
            if not LossFactory.get(loss_key):
                fallback = "asymmetric" if is_multilabel else "focal"
                logger.warning(f"Loss '{loss_key}' not found. Using fallback '{fallback}'.")
                loss_key = fallback
            
            # Prepare arguments based on loss type (Abstraction handling)
            loss_kwargs = {}
            if loss_key in ["focal", "adaptive_focal"]:
                loss_kwargs["gamma"] = params.get("gamma", 2.0)
                loss_kwargs["label_smoothing"] = params.get("label_smoothing", 0.0)
            
            elif loss_key in ["asymmetric", "asl"]:
                loss_kwargs["gamma_neg"] = params.get("gamma_neg", 4.0)
                loss_kwargs["gamma_pos"] = params.get("gamma_pos", 1.0)
                loss_kwargs["clip"] = params.get("asl_clip", 0.05)

            elif loss_key in ["binary_focal"]:
                 loss_kwargs["pos_weight"] = class_pos_weights.get(col) if class_pos_weights else None

            # Instantiate
            loss_cls = LossFactory.get(loss_key)
            criterions[col] = loss_cls(**loss_kwargs).to(device)
            
        return criterions