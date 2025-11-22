
import pytest
from unittest.mock import patch, MagicMock
from src.cfg.model_configs import get_model_config

@pytest.fixture
def mock_tomli_load():
    with patch("src.cfg.model_configs.tomli.load") as mock_load:
        yield mock_load

@pytest.fixture
def mock_open_file():
    with patch("builtins.open", new_callable=MagicMock) as mock_open:
        yield mock_open

@pytest.fixture
def mock_path_exists():
    with patch("pathlib.Path.exists") as mock_exists:
        mock_exists.return_value = True
        yield mock_exists

def test_get_model_config_success(mock_tomli_load, mock_open_file, mock_path_exists):
    # Setup mock return values
    # First call is global config, second is models config
    mock_tomli_load.side_effect = [
        { # Global config
            "hyperparameters": {"batch_size": 32, "lr": 0.001},
            "project": {"multi_label_cols": ["se"]}
        },
        { # Models config
            "test_model": {
                "features": ["f1"],
                "targets": ["t1"],
                "params": {"lr": 0.01} # Override
            }
        }
    ]

    config = get_model_config("test_model")

    assert config["batch_size"] == 32
    assert config["lr"] == 0.01 # Should be overridden
    assert config["features"] == ["f1"]
    assert config["targets"] == ["t1"]
    assert "multi_label_cols" in config # merged from project

def test_get_model_config_missing_model(mock_tomli_load, mock_open_file, mock_path_exists):
    mock_tomli_load.side_effect = [
        {}, # Global
        {"other_model": {}} # Models
    ]

    with pytest.raises(ValueError, match="not defined"):
        get_model_config("missing_model")

def test_get_model_config_missing_required_fields(mock_tomli_load, mock_open_file, mock_path_exists):
    mock_tomli_load.side_effect = [
        {},
        {"bad_model": {"params": {}}} # Missing features/targets
    ]

    with pytest.raises(ValueError, match="missing 'features' or 'targets'"):
        get_model_config("bad_model")
