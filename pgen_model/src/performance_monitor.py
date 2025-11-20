"""
Performance monitoring utilities for training optimization.

This module provides tools to monitor and log performance metrics during training,
including GPU memory usage, data loading throughput, and training speed.
"""

import logging
import time
from typing import Dict, Optional
import torch
import psutil

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor training performance metrics.

    Tracks GPU memory, CPU usage, data loading time, and training throughput.
    Provides summary statistics and identifies potential bottlenecks.
    """

    def __init__(self, enabled: bool = True):
        """
        Initialize performance monitor.

        Args:
            enabled: If False, all monitoring is disabled (zero overhead)
        """
        self.enabled = enabled
        self.metrics = {
            "data_loading_time": [],
            "forward_time": [],
            "backward_time": [],
            "batch_times": [],
            "gpu_memory_allocated": [],
            "gpu_memory_reserved": [],
        }
        self.start_time = None
        self.batch_start_time = None

    def start_batch(self):
        """Mark the start of a batch."""
        if not self.enabled:
            return
        self.batch_start_time = time.perf_counter()

    def record_data_loading(self):
        """Record data loading time for current batch."""
        if not self.enabled or self.batch_start_time is None:
            return
        elapsed = time.perf_counter() - self.batch_start_time
        self.metrics["data_loading_time"].append(elapsed)

    def record_forward(self):
        """Record forward pass time."""
        if not self.enabled or self.batch_start_time is None:
            return
        elapsed = time.perf_counter() - self.batch_start_time
        self.metrics["forward_time"].append(elapsed)

    def record_backward(self):
        """Record backward pass time."""
        if not self.enabled or self.batch_start_time is None:
            return
        elapsed = time.perf_counter() - self.batch_start_time
        self.metrics["backward_time"].append(elapsed)

    def record_batch_end(self):
        """Record end of batch and GPU memory usage."""
        if not self.enabled or self.batch_start_time is None:
            return

        elapsed = time.perf_counter() - self.batch_start_time
        self.metrics["batch_times"].append(elapsed)

        if torch.cuda.is_available():
            self.metrics["gpu_memory_allocated"].append(
                torch.cuda.memory_allocated() / 1024**3  # GB
            )
            self.metrics["gpu_memory_reserved"].append(
                torch.cuda.memory_reserved() / 1024**3  # GB
            )

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of monitored metrics.

        Returns:
            Dictionary with mean values for each metric
        """
        if not self.enabled:
            return {}

        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"{key}_mean"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)

        # Add system metrics
        summary["cpu_percent"] = psutil.cpu_percent()
        summary["ram_percent"] = psutil.virtual_memory().percent

        if torch.cuda.is_available():
            summary["gpu_memory_peak_gb"] = torch.cuda.max_memory_allocated() / 1024**3

        return summary

    def log_summary(self):
        """Log performance summary."""
        if not self.enabled:
            return

        summary = self.get_summary()
        logger.info("=== Performance Summary ===")

        if "batch_times_mean" in summary:
            logger.info(f"Avg batch time: {summary['batch_times_mean']:.4f}s")
            logger.info(
                f"Throughput: {1.0 / summary['batch_times_mean']:.2f} batches/s"
            )

        if "data_loading_time_mean" in summary:
            logger.info(f"Avg data loading: {summary['data_loading_time_mean']:.4f}s")

        if "gpu_memory_peak_gb" in summary:
            logger.info(f"Peak GPU memory: {summary['gpu_memory_peak_gb']:.2f} GB")

        logger.info(f"CPU usage: {summary.get('cpu_percent', 0):.1f}%")
        logger.info(f"RAM usage: {summary.get('ram_percent', 0):.1f}%")
        logger.info("=" * 30)

    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def estimate_optimal_batch_size(
    model: torch.nn.Module,
    sample_input: Dict[str, torch.Tensor],
    max_batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> int:
    """
    Estimate optimal batch size for given model and GPU memory.

    Performs binary search to find largest batch size that fits in memory
    with 20% safety margin.

    Args:
        model: PyTorch model to test
        sample_input: Sample batch input dictionary
        max_batch_size: Maximum batch size to try
        device: Device to run on (defaults to CUDA if available)

    Returns:
        Recommended batch size
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot estimate batch size")
        return 32  # Default fallback

    model = model.to(device)
    model.eval()

    # Binary search for max batch size
    low, high = 1, max_batch_size
    optimal_batch_size = 1

    while low <= high:
        mid = (low + high) // 2
        torch.cuda.empty_cache()

        try:
            # Create batch of size mid
            batch = {
                key: val.repeat(mid, *([1] * (val.dim() - 1))).to(device)
                for key, val in sample_input.items()
            }

            # Try forward pass
            with torch.no_grad():
                _ = model(**batch)

            # Success - try larger batch
            optimal_batch_size = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                # OOM - try smaller batch
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise

    # Apply 20% safety margin
    recommended = int(optimal_batch_size * 0.8)
    logger.info(
        f"Recommended batch size: {recommended} (max tested: {optimal_batch_size})"
    )

    return max(1, recommended)
