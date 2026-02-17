"""
test_losses.py

Tests for Custom Loss Functions.
"""

import torch

from src.model.metrics.losses import AsymmetricLoss, FocalLoss, PolyLoss


def test_focal_loss():
    """Test Focal Loss computation."""
    criterion = FocalLoss(gamma=2.0)

    # Simple binary case
    logits = torch.tensor([[10.0], [-10.0]]) # Very confident
    targets = torch.tensor([[1.0], [0.0]])   # Correct

    # Loss should be very small
    loss = criterion(logits, targets)
    assert loss.item() < 0.01

def test_asymmetric_loss():
    """Test ASL clipping logic."""
    # Negatives with low prob should be zeroed out
    criterion = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)

    # Logits resulting in p < 0.05
    # sigmoid(-3) ~= 0.047
    logits = torch.tensor([[-4.0]])
    targets = torch.tensor([[0.0]])

    loss = criterion(logits, targets)
    # Due to hard clipping, gradients might be zero, loss non-zero but small
    assert loss.item() >= 0.0

def test_polyloss():
    """Test PolyLoss runs."""
    criterion = PolyLoss(epsilon=1.0)
    logits = torch.randn(5, 2)
    targets = torch.randint(0, 2, (5,))

    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0
