"""
conftest.py

Pytest fixtures for Pharmagen.
Provides shared resources like mock dataframes and configurations.
"""

import pandas as pd
import pytest
from src.model.architecture.deep_fm import ModelConfig


@pytest.fixture
def sample_df():
    """Creates a small dummy dataset for testing."""
    data = {
        "drug": ["Aspirin", "Ibuprofen", "Paracetamol", "UnknownDrug"],
        "gene": ["CYP2D6", "CYP2C9", "CYP2D6", "CYP3A4"],
        "variant": ["rs123", "rs456", "rs123", "rs789"],
        # Multi-label column (pipe separated)
        "phenotype": ["Toxicity|Headache", "Efficacy", "Toxicity", "Efficacy|Nausea"],
        # Single label target
        "outcome": ["Adverse", "Success", "Adverse", "Success"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def mock_dims():
    """Mock dimensions usually provided by the Processor."""
    return {
        "n_features": {
            "drug": 10,
            "gene": 5,
            "variant": 20
        },
        "target_dims": {
            "phenotype": 4, # Multi-label
            "outcome": 2    # Multi-class
        }
    }

@pytest.fixture
def model_config(mock_dims):
    """A valid ModelConfig based on mock_dims."""
    return ModelConfig(
        n_features=mock_dims["n_features"],
        target_dims=mock_dims["target_dims"],
        embedding_dim=16,
        hidden_dim=32,
        n_layers=1,
        use_transformer=True,
        num_attn_layers=1,
        attn_heads=2,
        attn_dim_feedforward=32,
        fm_hidden_dim=8
    )
