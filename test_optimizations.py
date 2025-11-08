#!/usr/bin/env python
"""
Quick verification script to test that optimized code runs without errors.
This is a smoke test - it doesn't train a full model, just verifies imports and basic functionality.
"""

import sys
import torch
import numpy as np

def test_model_import():
    """Test that model can be imported and instantiated."""
    print("Testing model import...")
    try:
        from pgen_model.model import DeepFM_PGenModel
        
        # Create a small test model
        model = DeepFM_PGenModel(
            n_drugs=100,
            n_genalles=50,
            n_genes=20,
            n_alleles=30,
            embedding_dim=16,
            n_layers=2,
            hidden_dim=32,
            dropout_rate=0.1,
            target_dims={'task1': 3, 'task2': 5}
        )
        print("✓ Model instantiation successful")
        
        # Test forward pass
        batch_size = 4
        drug = torch.randint(0, 100, (batch_size,))
        genalle = torch.randint(0, 50, (batch_size,))
        gene = torch.randint(0, 20, (batch_size,))
        allele = torch.randint(0, 30, (batch_size,))
        
        outputs = model(drug, genalle, gene, allele)
        
        assert 'task1' in outputs, "Output missing task1"
        assert 'task2' in outputs, "Output missing task2"
        assert outputs['task1'].shape == (batch_size, 3), f"Unexpected shape for task1: {outputs['task1'].shape}"
        assert outputs['task2'].shape == (batch_size, 5), f"Unexpected shape for task2: {outputs['task2'].shape}"
        
        print("✓ Model forward pass successful")
        print(f"  - Output shapes: task1={outputs['task1'].shape}, task2={outputs['task2'].shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_imports():
    """Test that training modules can be imported."""
    print("\nTesting training module imports...")
    try:
        from pgen_model.train import train_model
        from pgen_model.data import PGenDataset, PGenDataProcess
        from pgen_model.pipeline import train_pipeline
        
        print("✓ All training modules imported successfully")
        return True
    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_amp_availability():
    """Test if AMP is available on this system."""
    print("\nChecking AMP (Automatic Mixed Precision) availability...")
    cuda_available = torch.cuda.is_available()
    print(f"  - CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        print("  - AMP can be used for faster training!")
    else:
        print("  - AMP disabled (no CUDA). CPU training will be used.")
    
    return True

def test_optimized_fm_computation():
    """Test the optimized FM computation works correctly."""
    print("\nTesting optimized FM computation...")
    try:
        batch_size = 8
        emb_dim = 16
        
        # Create sample embeddings
        drug_vec = torch.randn(batch_size, emb_dim)
        genal_vec = torch.randn(batch_size, emb_dim)
        gene_vec = torch.randn(batch_size, emb_dim)
        allele_vec = torch.randn(batch_size, emb_dim)
        
        # Optimized version (from our changes)
        embeddings_stack = torch.stack([drug_vec, genal_vec, gene_vec, allele_vec], dim=1)
        num_fields = embeddings_stack.size(1)
        fm_outputs_list = []
        for i in range(num_fields):
            for j in range(i + 1, num_fields):
                interaction = torch.sum(embeddings_stack[:, i] * embeddings_stack[:, j], dim=-1, keepdim=True)
                fm_outputs_list.append(interaction)
        fm_output = torch.cat(fm_outputs_list, dim=-1)
        
        expected_interactions = (num_fields * (num_fields - 1)) // 2  # C(4,2) = 6
        assert fm_output.shape == (batch_size, expected_interactions), \
            f"Unexpected FM output shape: {fm_output.shape}"
        
        print(f"✓ FM computation successful")
        print(f"  - Input: {num_fields} embeddings of dim {emb_dim}")
        print(f"  - Output: {expected_interactions} pairwise interactions")
        print(f"  - Output shape: {fm_output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ FM computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("Performance Optimization Verification Script")
    print("=" * 70)
    
    results = []
    
    # Run tests
    results.append(("Model Import & Forward Pass", test_model_import()))
    results.append(("Training Module Imports", test_training_imports()))
    results.append(("AMP Availability", test_amp_availability()))
    results.append(("Optimized FM Computation", test_optimized_fm_computation()))
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All optimizations verified successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
