
import pytest
from unittest.mock import patch, MagicMock
import torch
import pandas as pd
import numpy as np
from src.predict import PGenPredictor, UNKNOWN_TOKEN

@pytest.fixture
def mock_components():
    with patch("src.predict.get_model_config") as mock_config, \
         patch("src.predict.joblib.load") as mock_joblib_load, \
         patch("src.predict.torch.load") as mock_torch_load, \
         patch("src.predict.DeepFM_PGenModel") as mock_model_cls, \
         patch("src.predict.Path.exists") as mock_exists:
        
        mock_exists.return_value = True
        yield {
            "config": mock_config,
            "joblib": mock_joblib_load,
            "torch_load": mock_torch_load,
            "model_cls": mock_model_cls
        }

def test_predictor_initialization(mock_components):
    mocks = mock_components
    
    # Setup Config
    mocks["config"].return_value = {
        "features": ["feat1"],
        "targets": ["target1"],
        "params": {
            "embedding_dim": 10,
            "hidden_dim": 10,
            "dropout_rate": 0.1,
            "n_layers": 1
        }
    }
    
    # Setup Encoders
    mock_enc = MagicMock()
    mock_enc.classes_ = np.array(["A", "B"])
    mocks["joblib"].return_value = {"feat1": mock_enc, "target1": mock_enc}
    
    # Init Predictor
    predictor = PGenPredictor("test_model", device="cpu")
    
    assert predictor.model_name == "test_model"
    assert predictor.device.type == "cpu"
    mocks["model_cls"].assert_called_once()
    predictor.model.load_state_dict.assert_called_once()
    predictor.model.eval.assert_called_once()

def test_predict_single(mock_components):
    mocks = mock_components
    
    # 1. Setup (copy from init test)
    mocks["config"].return_value = {"features": ["feat1"], "targets": ["target1"], "params": {}}
    
    mock_enc = MagicMock()
    mock_enc.classes_ = np.array(["A", "B"])
    # Mock transform to return a specific index
    mock_enc.transform.return_value = np.array([0])
    mock_enc.inverse_transform.return_value = np.array(["A"])
    
    mocks["joblib"].return_value = {"feat1": mock_enc, "target1": mock_enc}
    
    # Mock Model Output
    mock_model_instance = mocks["model_cls"].return_value
    # Return a dict of tensors (logits)
    mock_model_instance.return_value = {"target1": torch.tensor([[2.0, -2.0]])} # Class 0 wins
    
    predictor = PGenPredictor("test_model", device="cpu")
    
    # 2. Execute
    input_data = {"feat1": "A"}
    result = predictor.predict_single(input_data)
    
    # 3. Assert
    assert result is not None
    assert result["target1"] == "A"
    mock_enc.transform.assert_called() # Called during input prep
    
def test_predict_file(mock_components):
    mocks = mock_components
    
    # Setup similar to single but ensure vectorized transform works
    mocks["config"].return_value = {"features": ["feat1"], "targets": ["target1"], "params": {}}
    
    mock_enc = MagicMock()
    mock_enc.classes_ = np.array(["A", "B"])
    # Vectorized transform
    mock_enc.transform.return_value = np.array([0, 1]) 
    mock_enc.inverse_transform.return_value = np.array(["A", "B"])
    
    mocks["joblib"].return_value = {"feat1": mock_enc, "target1": mock_enc}
    
    # Mock Model
    mock_model_instance = mocks["model_cls"].return_value
    # Batch output for 2 samples
    mock_model_instance.return_value = {"target1": torch.tensor([[2.0, -2.0], [-2.0, 2.0]])}
    
    predictor = PGenPredictor("test_model", device="cpu")
    
    # Create dummy CSV
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({"feat1": ["A", "B"]})
        
        results = predictor.predict_file("dummy.csv", batch_size=2)
        
        assert len(results) == 2
        assert results[0]["target1"] == "A"
        assert results[1]["target1"] == "B"

