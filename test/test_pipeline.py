
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.pipeline import train_pipeline

@pytest.fixture
def mock_dependencies():
    with patch("src.pipeline.get_model_config") as mock_config, \
         patch("src.pipeline.load_and_prep_dataset") as mock_load_data, \
         patch("src.pipeline.PGenDataProcess") as mock_processor, \
         patch("src.pipeline.DeepFM_PGenModel") as mock_model, \
         patch("src.pipeline.train_model") as mock_train, \
         patch("src.pipeline.save_model") as mock_save, \
         patch("src.pipeline.joblib.dump") as mock_dump, \
         patch("src.pipeline.Path") as mock_path:
        
        yield {
            "config": mock_config,
            "load_data": mock_load_data,
            "processor": mock_processor,
            "model": mock_model,
            "train": mock_train,
            "save": mock_save,
            "dump": mock_dump,
            "path": mock_path
        }

def test_train_pipeline_execution(mock_dependencies):
    # Setup Mocks
    mocks = mock_dependencies
    
    # 1. Config
    mocks["config"].return_value = {
        "features": ["gene_id"],
        "targets": ["outcome"],
        "cols": ["gene_id", "outcome"],
        "params": {
            "batch_size": 16,
            "embedding_dim": 8,
            "hidden_dim": 16,
            "dropout_rate": 0.1,
            "n_layers": 1
        }
    }

    # 2. Data Loading
    dummy_df = pd.DataFrame({
        "gene_id": ["G1"] * 10, 
        "outcome": [0] * 10,
        "stratify_col": [0] * 10
    })
    mocks["load_data"].return_value = dummy_df

    # 3. Processor
    mock_proc_instance = mocks["processor"].return_value
    mock_proc_instance.encoders = {"gene_id": MagicMock(classes_=[_G1_]), "outcome": MagicMock(classes_=[0, 1])}
    
    # 4. Train Model Return
    mocks["train"].return_value = (0.5, {"outcome": 0.9}, {"outcome": 0.1})

    # Execute Pipeline
    train_pipeline(
        csv_path="dummy.csv",
        model_name="test_model",
        epochs=1,
        patience=1
    )

    # Assertions
    mocks["config"].assert_called_once_with("test_model")
    mocks["load_data"].assert_called_once()
    mocks["processor"].assert_called_once()
    mock_proc_instance.fit.assert_called_once()
    mocks["model"].assert_called_once()
    mocks["train"].assert_called_once()
    mocks["save"].assert_called_once()
    mocks["dump"].assert_called_once()
