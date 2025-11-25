import pytest
from unittest.mock import patch, MagicMock
from src.cfg.manager import get_model_config

@pytest.fixture
def mock_tomli_load():
    with patch("src.cfg.manager.tomllib.load") as mock_load:
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
    # Order: paths.toml, config.toml (Global), models.toml (Models)
    mock_tomli_load.side_effect = [
        {"base": {"data": "data"}}, # paths.toml
        { # Global config.toml
            "params": {"batch_size": 32, "learning_rate": 0.001},
            "project": {"multi_label_cols": ["se"]}
        },
        { # Models config.toml
            "test_model": {
                "features": ["f1"],
                "targets": ["t1"],
                "params": {"learning_rate": 0.01} # Override
            }
        }
    ]

    config = get_model_config("test_model")

    assert config["batch_size"] == 32
    assert config["learning_rate"] == 0.01 # Should be overridden
    assert config["features"] == ["f1"]
    assert config["targets"] == ["t1"]
    assert "multi_label_cols" in config # merged from project

def test_get_model_config_missing_model(mock_tomli_load, mock_open_file, mock_path_exists):
    mock_tomli_load.side_effect = [
        {"base": {"data": "data"}},
        {}, # Global
        {"other_model": {}} # Models
    ]

    with pytest.raises(ValueError, match="not found in models.toml"):
        get_model_config("missing_model")

def test_get_model_config_missing_required_fields(mock_tomli_load, mock_open_file, mock_path_exists):
    mock_tomli_load.side_effect = [
        {"base": {"data": "data"}},
        {},
        {"bad_model": {"params": {}}} # Missing features/targets
    ]

    with pytest.raises(ValueError, match="requires \['features', 'targets'\]"):
        get_model_config("bad_model")