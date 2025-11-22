
import pandas as pd
import numpy as np
import pytest
import torch
from src.data import PGenDataProcess, PGenDataset
from src.utils.data import UNKNOWN_TOKEN

@pytest.fixture
def sample_data():
    data = {
        "gene_id": ["G1", "G2", "G3", "G1"],
        "drug_id": ["D1", "D2", "D1", "D3"],
        "side_effects": ["SE1|SE2", "SE2", "SE1|SE3", "SE1"],  # Multi-label
        "target_outcome": [0, 1, 0, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def feature_cols():
    return ["gene_id", "drug_id"]

@pytest.fixture
def target_cols():
    return ["target_outcome"]

@pytest.fixture
def multi_label_cols():
    return ["side_effects"]

class TestPGenDataProcess:
    def test_fit_transform(self, sample_data, feature_cols, target_cols, multi_label_cols):
        processor = PGenDataProcess(
            feature_cols=feature_cols,
            target_cols=target_cols,
            multi_label_cols=multi_label_cols
        )
        
        # Test Fit
        processor.fit(sample_data)
        assert "gene_id" in processor.encoders
        assert "drug_id" in processor.encoders
        assert "side_effects" in processor.encoders
        
        # Test Transform
        transformed_df = processor.transform(sample_data)
        
        # Check if columns are transformed
        assert transformed_df["gene_id"].dtype == int or np.issubdtype(transformed_df["gene_id"].dtype, np.integer)
        assert transformed_df["drug_id"].dtype == int or np.issubdtype(transformed_df["drug_id"].dtype, np.integer)
        
        # Check MultiLabel column (should be object/list of arrays or similar structure depending on implementation)
        # based on code: df_out[col] = pd.Series(list(encoded_data), ...)
        assert isinstance(transformed_df["side_effects"].iloc[0], (np.ndarray, list))
        
    def test_transform_unknown_values(self, sample_data, feature_cols, target_cols):
        processor = PGenDataProcess(feature_cols=feature_cols, target_cols=target_cols)
        processor.fit(sample_data)
        
        new_data = pd.DataFrame({
            "gene_id": ["G_NEW"],
            "drug_id": ["D1"],
            "target_outcome": [0]
        })
        
        transformed = processor.transform(new_data)
        
        # Check that unknown token logic handles it (mapped to UNKNOWN_TOKEN index if it exists or handle gracefully)
        # The code adds UNKNOWN_TOKEN to classes during fit if not present.
        unknown_idx = processor.encoders["gene_id"].transform([UNKNOWN_TOKEN])[0]
        assert transformed["gene_id"].iloc[0] == unknown_idx

class TestPGenDataset:
    def test_dataset_structure(self, sample_data, feature_cols, target_cols, multi_label_cols):
        # Preprocess first
        processor = PGenDataProcess(
            feature_cols=feature_cols,
            target_cols=target_cols,
            multi_label_cols=multi_label_cols
        )
        processor.fit(sample_data)
        processed_df = processor.transform(sample_data)
        
        dataset = PGenDataset(
            processed_df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            multi_label_cols=set(multi_label_cols)
        )
        
        assert len(dataset) == len(sample_data)
        
        # Get item
        item = dataset[0]
        
        # Check keys
        for col in feature_cols:
            assert col in item
        for col in target_cols:
            assert col in item
        for col in multi_label_cols:
            assert col in item
            
        # Check Types
        assert isinstance(item["gene_id"], torch.Tensor)
        assert item["gene_id"].dtype == torch.long
        
        assert isinstance(item["side_effects"], torch.Tensor)
        assert item["side_effects"].dtype == torch.float32  # Multi-label usually float for BCE logic or embedding sum
