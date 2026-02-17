"""
test_modeling.py

Tests for PharmagenDeepFM Architecture.
"""

import torch

from src.model.architecture.deep_fm import PharmagenDeepFM


def test_model_initialization(model_config):
    """Ensure model builds without error."""
    model = PharmagenDeepFM(model_config)
    assert model.deep_mlp is not None
    assert model.transformer is not None
    assert len(model.heads) == 2

def test_forward_pass(model_config):
    """Test forward pass output shapes."""
    model = PharmagenDeepFM(model_config)

    # Create Dummy Batch
    batch_size = 5
    inputs = {
        "drug": torch.randint(0, 10, (batch_size,)),
        "gene": torch.randint(0, 5, (batch_size,)),
        "variant": torch.randint(0, 20, (batch_size,))
    }

    outputs = model(inputs)

    # Check outputs
    assert "phenotype" in outputs
    assert "outcome" in outputs

    # Check dimensions
    # Phenotype -> 4 classes (from conftest)
    assert outputs["phenotype"].shape == (batch_size, 4)
    # Outcome -> 2 classes
    assert outputs["outcome"].shape == (batch_size, 2)

def test_backward_pass(model_config):
    """Test that gradients flow (no detached graphs)."""
    model = PharmagenDeepFM(model_config)

    inputs = {
        "drug": torch.randint(0, 10, (2,)),
        "gene": torch.randint(0, 5, (2,)),
        "variant": torch.randint(0, 20, (2,))
    }

    outputs = model(inputs)
    loss = outputs["outcome"].sum()
    loss.backward()

    # Check if embedding weights have grad
    assert model.embeddings["drug"].weight.grad is not None
    assert model.deep_mlp.network[0].weight.grad is not None
