"""
test_data.py

Tests for DataHandler, Processor and Dataset.
"""
import numpy as np
import pandas as pd
import pytest
import torch
from src.data_handler import DataConfig, PGenDataset, PGenProcessor


@pytest.fixture
def test_processor_fit_transform(sample_df):
    """Test if PGenProcessor correctly encodes scalars and multi-labels."""
    config = {
        "features": ["drug", "gene", "variant"],
        "targets": ["outcome"],
    }
    multi_label_cols = ["phenotype"]

    processor = PGenProcessor(config, multi_label_cols)
    processor.fit(sample_df)

    # Check if encoders are created
    assert "drug" in processor.encoders
    assert "phenotype" in processor.encoders

    # Check Transform
    tensors = processor.transform(sample_df)

    # Scalar check
    assert isinstance(tensors["drug"], torch.Tensor)
    assert tensors["drug"].dtype == torch.long
    assert len(tensors["drug"]) == len(sample_df)

    # Multi-label check
    assert isinstance(tensors["phenotype"], torch.Tensor)
    assert tensors["phenotype"].dtype == torch.float32
    assert tensors["phenotype"].dim() == 2

    # Outcome check
    assert "outcome" in tensors

def test_unknown_handling(sample_df):
    """Test robust handling of unseen values."""
    config = {"features": ["drug"], "targets": []}
    processor = PGenProcessor(config)
    processor.fit(sample_df)

    # Create new DF with unseen drug
    new_df = pd.DataFrame({"drug": ["NewMysteriousDrug"]})

    tensors = processor.transform(new_df)

    enc = processor.encoders["drug"]
    unknown_idx = np.searchsorted(enc.classes_, "__UNKNOWN__")

    assert tensors["drug"][0].item() == unknown_idx

def test_dataset_item(sample_df):
    """Test PGenDataset __getitem__ retrieval."""
    config = {"features": ["drug"], "targets": ["outcome"]}
    processor = PGenProcessor(config)
    processor.fit(sample_df)
    data = processor.transform(sample_df)

    ds = PGenDataset(
        data,
        features=["drug"],
        targets=["outcome"],
        multi_label_cols=set()
    )

    assert len(ds) == 4

    sample_x, sample_y = ds[0]

    assert "drug" in sample_x
    assert "outcome" in sample_y
    assert torch.is_tensor(sample_x["drug"])
