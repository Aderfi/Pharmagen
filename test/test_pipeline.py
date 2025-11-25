import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.pipeline import train_pipeline

@pytest.fixture
def mock_dependencies():
    with patch("src.pipeline.get_model_config") as mock_config, \
         patch("src.pipeline.load_dataset") as mock_load_data, \
         patch("src.pipeline.PGenProcessor") as mock_processor, \
         patch("src.pipeline.create_model") as mock_create_model, \
         patch("src.pipeline.PGenTrainer") as mock_trainer_cls, \
         patch("src.pipeline.joblib.dump") as mock_dump, \
         patch("src.pipeline.save_json") as mock_save_json, \
         patch("src.pipeline.DIRS") as mock_dirs:
        
        # Mock directory paths
        mock_dirs.__getitem__.return_value = MagicMock()
        
        yield {
            "config": mock_config,
            "load_data": mock_load_data,
            "processor": mock_processor,
            "create_model": mock_create_model,
            "trainer_cls": mock_trainer_cls,
            "dump": mock_dump,
            "save_json": mock_save_json
        }

def test_train_pipeline_execution(mock_dependencies):
    # Setup Mocks
    mocks = mock_dependencies
    
    # 1. Config
    mocks["config"].return_value = {
        "features": ["gene_id"],
        "targets": ["outcome"],
        "params": {
            "batch_size": 16,
            "embedding_dim": 8,
            "hidden_dim": 16,
            "dropout_rate": 0.1,
            "n_layers": 1,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "early_stopping_patience": 1
        }
    }

    # 2. Data Loading
    dummy_df = pd.DataFrame({
        "gene_id": ["G1"] * 10, 
        "outcome": ["0"] * 10,
        "_stratify": ["0"] * 10
    })
    mocks["load_data"].return_value = dummy_df

    # 3. Processor
    mock_proc_instance = mocks["processor"].return_value
    mock_proc_instance.encoders = {"gene_id": MagicMock(classes_=[_G1_]), "outcome": MagicMock(classes_=[0, 1])}
    
    # 4. Trainer
    mock_trainer_instance = mocks["trainer_cls"].return_value
    mock_trainer_instance.fit.return_value = 0.5 # Best loss

    # Execute Pipeline
    train_pipeline(
        model_name="test_model",
        csv_path="dummy.csv",
        epochs=1
    )

    # Assertions
    mocks["config"].assert_called_once_with("test_model")
    mocks["load_data"].assert_called_once()
    mocks["processor"].assert_called_once()
    mock_proc_instance.fit.assert_called_once()
    mocks["create_model"].assert_called_once()
    mocks["trainer_cls"].assert_called_once()
    mock_trainer_instance.fit.assert_called_once()
    mocks["dump"].assert_called_once() # Save encoders
    mocks["save_json"].assert_called_once() # Save config report