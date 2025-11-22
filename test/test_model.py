
import pytest
import torch
from src.model import DeepFM_PGenModel

@pytest.fixture
def model_params():
    return {
        "n_features": {"gene_id": 100, "drug_id": 50},
        "target_dims": {"outcome": 1, "side_effects": 10},
        "embedding_dim": 16,
        "hidden_dim": 32,
        "dropout_rate": 0.1,
        "n_layers": 2,
        "attention_dim_feedforward": 64,
        "attention_dropout": 0.1,
        "num_attention_layers": 1,
        "use_batch_norm": True,
        "use_layer_norm": False,
        "activation_function": "relu",
        "fm_dropout": 0.1,
        "fm_hidden_layers": 1,
        "fm_hidden_dim": 16,
        "embedding_dropout": 0.1
    }

class TestDeepFM_PGenModel:
    def test_model_initialization(self, model_params):
        model = DeepFM_PGenModel(**model_params)
        assert isinstance(model, DeepFM_PGenModel)
        
    def test_forward_pass(self, model_params):
        model = DeepFM_PGenModel(**model_params)
        
        batch_size = 4
        # Create dummy inputs
        inputs = {
            "gene_id": torch.randint(0, model_params["n_features"]["gene_id"], (batch_size,)),
            "drug_id": torch.randint(0, model_params["n_features"]["drug_id"], (batch_size,))
        }
        
        outputs = model(inputs)
        
        # Check output keys
        assert "outcome" in outputs
        assert "side_effects" in outputs
        
        # Check output shapes
        assert outputs["outcome"].shape == (batch_size, model_params["target_dims"]["outcome"])
        assert outputs["side_effects"].shape == (batch_size, model_params["target_dims"]["side_effects"])

    def test_forward_pass_missing_feature(self, model_params):
        model = DeepFM_PGenModel(**model_params)
        inputs = {
            "gene_id": torch.randint(0, 10, (2,))
            # Missing 'drug_id'
        }
        
        with pytest.raises(ValueError, match="Features faltantes"):
            model(inputs)
