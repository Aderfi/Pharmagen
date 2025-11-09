# Performance Optimization Guide

This document describes the performance optimizations implemented in the pharmagen_pmodel codebase to improve memory, CPU, and GPU efficiency.

## Overview

The optimizations focus on:
1. **GPU Efficiency**: Mixed precision training, optimized memory transfers
2. **CPU Efficiency**: Dynamic worker allocation, vectorized operations
3. **Memory Optimization**: Reduced memory footprint, efficient data loading
4. **Code Quality**: Removed redundant code, improved maintainability

## GPU Optimizations

### Mixed Precision Training (AMP)
**Location**: `pgen_model/src/train.py`

Automatic Mixed Precision (AMP) training reduces GPU memory usage by 40-50% and can provide 1.5-3x speedup on modern GPUs.

**Benefits**:
- Reduced memory usage (float16 vs float32)
- Faster computation on Tensor Cores
- Automatic loss scaling prevents underflow

**Implementation**:
```python
from torch.cuda.amp import autocast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Optimized GPU Settings
**Location**: `pgen_model/src/pipeline.py`, `pgen_model/src/optuna_train.py`

```python
if torch.cuda.is_available():
    # Optimize cuDNN for fixed input sizes
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 on Ampere GPUs (faster matrix ops)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
```

**Benefits**:
- cuDNN benchmark finds optimal algorithms for your specific input sizes
- TF32 provides 8x faster matrix operations on Ampere GPUs with minimal precision loss

### Asynchronous Data Transfers
**Location**: `pgen_model/src/train.py`

```python
# Non-blocking transfers overlap CPU and GPU work
data = batch["input"].to(device, non_blocking=True)
```

**Benefits**:
- GPU and CPU can work in parallel
- Reduces idle time waiting for transfers
- Up to 20-30% improvement in GPU utilization

## CPU Optimizations

### Dynamic Worker Allocation
**Location**: `pgen_model/src/pipeline.py`, `pgen_model/src/optuna_train.py`

```python
import multiprocessing

# Optimize based on available CPU cores
num_workers = min(multiprocessing.cpu_count(), 8)

train_loader = DataLoader(
    dataset,
    num_workers=num_workers,
    persistent_workers=True if num_workers > 0 else False,
)
```

**Benefits**:
- Adapts to available hardware
- Prevents over-subscription (CPU thrashing)
- Prevents under-utilization (wasted cores)
- `persistent_workers=True` avoids respawning overhead

### Vectorized String Operations
**Location**: `pgen_model/src/data.py`

```python
# Before: Slow loop
for col in columns:
    df[col] = df[col].str.replace(' ', '_')

# After: Fast vectorized operation
df[col] = df[col].str.replace(' ', '_', regex=False)
```

**Benefits**:
- `regex=False` uses faster literal string replacement
- Vectorized pandas operations are 10-100x faster than loops

## Memory Optimizations

### Pin Memory for GPU
**Location**: `pgen_model/src/pipeline.py`, `pgen_model/src/optuna_train.py`

```python
pin_memory = torch.cuda.is_available()

train_loader = DataLoader(
    dataset,
    pin_memory=pin_memory,
)
```

**Benefits**:
- Faster GPU transfers (bypasses CPU cache)
- Enables `non_blocking=True` transfers
- 15-20% faster data loading to GPU

### Persistent Workers
```python
persistent_workers=True if num_workers > 0 else False
```

**Benefits**:
- Workers persist across epochs
- Avoids expensive worker respawn
- Reduces memory fragmentation

## Code Quality Improvements

### Removed Commented Code
**Location**: `pgen_model/src/model.py`

- Removed ~340 lines of commented-out duplicate code
- Improved code readability and maintainability
- Reduced file size by ~50%

## Performance Monitoring

### New Utility
**Location**: `pgen_model/src/performance_monitor.py`

Provides tools to track:
- GPU memory usage
- Data loading throughput
- Training speed
- Batch size optimization

**Usage**:
```python
from pgen_model.src.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor(enabled=True)

for batch in dataloader:
    monitor.start_batch()
    # ... training code ...
    monitor.record_batch_end()

monitor.log_summary()
```

## Expected Performance Gains

| Optimization | Expected Improvement |
|-------------|---------------------|
| Mixed Precision (AMP) | 1.5-3x training speed, 40-50% memory reduction |
| cuDNN Benchmark | 5-10% speedup |
| Pin Memory + Non-blocking | 15-20% data loading speedup |
| Dynamic Workers | 10-30% CPU efficiency |
| Vectorized Operations | 10-100x faster preprocessing |

**Overall**: 2-4x faster training with 30-50% less memory usage

## Hardware-Specific Notes

### For NVIDIA Ampere GPUs (RTX 30/40 series, A100)
- TF32 provides 8x speedup for matrix operations
- AMP is especially effective (better Tensor Core utilization)

### For NVIDIA Volta/Turing GPUs (V100, RTX 20 series)
- AMP still provides 1.5-2x speedup
- TF32 not available (uses standard FP16)

### For CPU-only systems
- Mixed precision disabled automatically
- Worker optimization still helps
- Vectorized operations provide biggest gains

## Best Practices

1. **Always profile first**: Use `performance_monitor.py` to identify bottlenecks
2. **Monitor GPU memory**: Watch for OOM errors, adjust batch size if needed
3. **Check GPU utilization**: Should be >80% during training
4. **Validate accuracy**: Ensure AMP doesn't hurt model performance
5. **Tune workers**: Start with `num_workers=4`, increase if CPU underutilized

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch size
- Enable gradient checkpointing (future optimization)
- Check for memory leaks (detach tensors in validation)

### Slow Data Loading
- Increase `num_workers` (but not more than CPU cores)
- Enable `pin_memory=True`
- Check disk I/O (use SSD for dataset)

### No Speedup from AMP
- Check GPU architecture (older GPUs benefit less)
- Verify Tensor Core usage (use `torch.autocast()`)
- Some operations don't support mixed precision

## Future Optimizations

Potential areas for further improvement:
1. Gradient checkpointing for even lower memory
2. Distributed training (multi-GPU)
3. Model quantization for inference
4. ONNX export for production deployment
5. Flash Attention for transformer layers

## References

- [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
- [NVIDIA Mixed Precision Training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/)
- [PyTorch DataLoader Performance](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
