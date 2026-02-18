"""
test_performance_optimizations.py

Tests to verify correctness of performance optimizations.
"""
import re

import pandas as pd
import pytest

from src.data.etl_pipeline import normalize_drug_names, _SALT_PATTERN
from src.model.metrics.reporting import _generate_table_rows


def test_normalize_drug_names():
    """Test that pre-compiled regex pattern works correctly."""
    # Test various salt removals
    assert normalize_drug_names("aspirin sodium") == "aspirin"
    assert normalize_drug_names("warfarin sodium") == "warfarin"
    assert normalize_drug_names("Metformin Hydrochloride") == "metformin"
    assert normalize_drug_names("Sertraline HCl") == "sertraline"
    
    # Test edge cases
    assert normalize_drug_names(None) == "unknown"
    assert normalize_drug_names("") == ""
    assert normalize_drug_names("Unknown Drug") == "unknown drug"


def test_salt_pattern_compiled():
    """Verify that the regex pattern is pre-compiled."""
    # Check that _SALT_PATTERN is a compiled regex
    assert isinstance(_SALT_PATTERN, re.Pattern)
    
    # Verify it matches expected salts
    assert _SALT_PATTERN.search("hydrochloride")
    assert _SALT_PATTERN.search("sodium")
    assert _SALT_PATTERN.search("hcl")
    assert not _SALT_PATTERN.search("aspirin")


def test_generate_table_rows():
    """Test that optimized table row generation produces correct HTML."""
    # Create sample training history DataFrame
    df = pd.DataFrame({
        "epoch": [1, 2, 3],
        "train_loss": [0.5, 0.4, 0.3],
        "val_loss": [0.6, 0.5, 0.4],
        "val_acc_macro": [0.70, 0.75, 0.80],
        "val_acc_weighted": [0.72, 0.77, 0.82]
    })
    
    acc_cols = ["val_acc_macro", "val_acc_weighted"]
    result = _generate_table_rows(df, acc_cols)
    
    # Verify it's a string
    assert isinstance(result, str)
    
    # Verify it contains table rows (reversed order)
    assert "<tr>" in result
    assert "</tr>" in result
    
    # Verify last epoch appears first (reversed)
    assert "3</td>" in result.split("<tr>")[1]
    
    # Verify all epochs are present
    assert "1</td>" in result
    assert "2</td>" in result
    assert "3</td>" in result
    
    # Verify loss values are formatted correctly
    assert "0.3000" in result  # epoch 3 train_loss
    assert "0.4000" in result  # epoch 3 val_loss


def test_generate_table_rows_empty():
    """Test table generation with empty DataFrame."""
    df = pd.DataFrame({
        "epoch": [],
        "train_loss": [],
        "val_loss": []
    })
    
    result = _generate_table_rows(df, [])
    assert result == ""


@pytest.mark.parametrize("drug_input,expected", [
    ("Aspirin", "aspirin"),
    ("Metformin HCl", "metformin"),
    ("Warfarin Sodium", "warfarin"),
    ("Sertraline Hydrochloride", "sertraline"),
    ("Unknown", "unknown"),
])
def test_normalize_drug_names_parametrized(drug_input, expected):
    """Parametrized test for drug name normalization."""
    result = normalize_drug_names(drug_input)
    assert result == expected
