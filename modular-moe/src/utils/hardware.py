"""Hardware utilities for memory estimation and device management."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass
from typing import Optional, Literal

import torch


@dataclass
class MemoryEstimate:
    """Memory estimation results."""
    param_memory_gb: float
    optimizer_memory_gb: float
    gradient_memory_gb: float
    activation_memory_gb: float
    total_gpu_memory_gb: float
    total_cpu_memory_gb: float
    num_parameters: int
    phase: str
    trainable_parameters: int
    warnings: list[str]


def get_gpu_memory(device_id: int = 0) -> tuple[int, int]:
    """Get free and total GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0, 0
    
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_id}",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        free, total = map(int, result.stdout.strip().split(","))
        return free * 1024**2, total * 1024**2
    except Exception:
        # Fallback to PyTorch
        free, total = torch.cuda.mem_get_info(device_id)
        return free, total


def get_cpu_memory() -> tuple[int, int]:
    """Get available and total CPU memory in bytes."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return mem.available, mem.total
    except ImportError:
        return 0, 0


def estimate_model_memory(
    num_parameters: int,
    trainable_parameters: int,
    dtype: torch.dtype = torch.bfloat16,
    optimizer: str = "adamw",
    batch_size: int = 1,
    seq_len: int = 2048,
    hidden_dim: int = 2664,
    num_layers: int = 32,
    gradient_checkpointing: bool = False,
    phase: str = "joint",
    gpu_vram_gb: float = 12.0,
    safety_threshold: float = 0.9,
) -> MemoryEstimate:
    """Estimate memory requirements for model training.
    
    Args:
        num_parameters: Total model parameters
        trainable_parameters: Number of trainable parameters
        dtype: Training dtype
        optimizer: Optimizer type
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension
        num_layers: Number of layers
        gradient_checkpointing: Whether gradient checkpointing is enabled
        phase: Training phase - "expert", "router", "joint"
        gpu_vram_gb: GPU VRAM in GB
        safety_threshold: Safety threshold for VRAM usage (0.9 = 90%)
    """
    
    # Bytes per parameter
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float8_e4m3fn: 1,
        torch.float8_e5m2: 1,
    }.get(dtype, 2)
    
    # Parameter memory (all parameters loaded)
    param_memory = num_parameters * dtype_bytes
    
    # Optimizer memory (only for trainable params, FP32 for AdamW)
    if optimizer.lower() == "adamw":
        optimizer_memory = trainable_parameters * 2 * 4  # FP32 for optimizer states
    elif optimizer.lower() == "sgd":
        optimizer_memory = trainable_parameters * 4  # FP32 momentum
    else:
        optimizer_memory = trainable_parameters * 2 * 4
    
    # Gradient memory (only for trainable params)
    gradient_memory = trainable_parameters * dtype_bytes
    
    # Activation memory estimation
    if gradient_checkpointing:
        active_layers = max(1, int(num_layers**0.5))
    else:
        active_layers = num_layers
    
    activation_elements = batch_size * seq_len * hidden_dim * active_layers
    activation_memory = activation_elements * dtype_bytes * 4  # ~4x for intermediate activations
    
    # Temporary/router memory for MoE
    if phase in ["router", "joint"]:
        router_temp = num_layers * batch_size * seq_len * 32 * 4  # ~router logits
    else:
        router_temp = 0
    
    # Total
    total_gpu = param_memory + optimizer_memory + gradient_memory + activation_memory + router_temp
    total_cpu = param_memory * 2  # For checkpointing to CPU
    
    warnings = []
    estimated_vram_gb = total_gpu / 1024**3
    max_allowed = gpu_vram_gb * safety_threshold
    
    if estimated_vram_gb > max_allowed:
        warnings.append(f"Estimated VRAM ({estimated_vram_gb:.2f} GB) exceeds safety threshold ({max_allowed:.2f} GB of {gpu_vram_gb} GB)")
    if estimated_vram_gb > gpu_vram_gb:
        warnings.append(f"Estimated VRAM ({estimated_vram_gb:.2f} GB) EXCEEDS GPU VRAM ({gpu_vram_gb} GB) - OOM likely")
    
    return MemoryEstimate(
        param_memory_gb=param_memory / 1024**3,
        optimizer_memory_gb=optimizer_memory / 1024**3,
        gradient_memory_gb=gradient_memory / 1024**3,
        activation_memory_gb=activation_memory / 1024**3,
        total_gpu_memory_gb=total_gpu / 1024**3,
        total_cpu_memory_gb=total_cpu / 1024**3,
        num_parameters=num_parameters,
        phase=phase,
        trainable_parameters=trainable_parameters,
        warnings=warnings,
    )


def print_memory_estimate(estimate: MemoryEstimate) -> None:
    """Pretty print memory estimate."""
    print("\n" + "=" * 70)
    print(f"MEMORY ESTIMATION - Phase: {estimate.phase.upper()}")
    print("=" * 70)
    print(f"Total Parameters:    {estimate.num_parameters:,} ({estimate.num_parameters/1e9:.2f}B)")
    print(f"Trainable Parameters: {estimate.trainable_parameters:,} ({estimate.trainable_parameters/1e6:.2f}M)")
    print(f"Parameter memory:    {estimate.param_memory_gb:.2f} GB")
    print(f"Optimizer memory:    {estimate.optimizer_memory_gb:.2f} GB")
    print(f"Gradient memory:     {estimate.gradient_memory_gb:.2f} GB")
    print(f"Activation memory:   {estimate.activation_memory_gb:.2f} GB")
    if hasattr(estimate, 'router_temp_gb') and estimate.router_temp_gb > 0:
        print(f"Router temp memory:  {estimate.router_temp_gb:.2f} GB")
    print("-" * 70)
    print(f"Total GPU memory:    {estimate.total_gpu_memory_gb:.2f} GB")
    print(f"Total CPU memory:    {estimate.total_cpu_memory_gb:.2f} GB")
    print("-" * 70)
    for warning in estimate.warnings:
        print(f"[WARN] WARNING: {warning}")
    print("=" * 70)


def set_cpu_affinity(core_count: int = 1, nice_value: int = 10) -> bool:
    """Limit CPU usage by setting process affinity and nice value.
    
    Args:
        core_count: Number of CPU cores to use (1 = ~50% on dual-core, ~25% on quad-core)
        nice_value: Process nice value (0-19, higher = lower priority). 10 = low priority.
    
    Returns:
        True if throttling was applied, False otherwise.
    """
    try:
        import psutil
        p = psutil.Process()
        cpu_count = psutil.cpu_count()
        
        # Pin to first N cores
        available_cores = list(range(min(core_count, cpu_count)))
        p.cpu_affinity(available_cores)
        
        # Set nice value to lower priority (may fail on some platforms)
        if hasattr(p, 'nice'):
            try:
                p.nice(nice_value)
            except Exception:
                # On Windows, nice values are different; ignore if not supported
                pass
        
        return True
    except Exception:
        return False


def get_device(device_preference: str = "cuda") -> torch.device:
    """Get the best available device."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_deterministic(seed: int = 42) -> None:
    """Set deterministic behavior for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Environment variables
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    # PyTorch 1.12+ deterministic
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_git_revision() -> Optional[str]:
    """Get current git revision."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def get_system_info() -> dict:
    """Get system information."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = get_gpu_memory(i)
            info[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": total / 1024**3,
                "free_memory_gb": free / 1024**3,
                "compute_capability": f"{props.major}.{props.minor}",
            }
    
    cpu_avail, cpu_total = get_cpu_memory()
    info["cpu_memory_gb"] = cpu_total / 1024**3
    info["cpu_available_gb"] = cpu_avail / 1024**3
    
    return info


def run_cuda_test() -> dict:
    """Run a small CUDA test to verify GPU functionality."""
    results = {
        "cuda_available": torch.cuda.is_available(),
        "test_passed": False,
        "error": None,
    }
    
    if not torch.cuda.is_available():
        results["error"] = "CUDA not available"
        return results
    
    try:
        # Test basic operations
        x = torch.randn(100, 100, device="cuda")
        y = torch.randn(100, 100, device="cuda")
        z = torch.mm(x, y)
        
        # Test FP16
        x_fp16 = torch.randn(100, 100, device="cuda", dtype=torch.float16)
        y_fp16 = torch.randn(100, 100, device="cuda", dtype=torch.float16)
        z_fp16 = torch.mm(x_fp16, y_fp16)
        
        # Test backward pass
        x = torch.randn(10, 10, device="cuda", requires_grad=True)
        y = x * 2
        loss = y.sum()
        loss.backward()
        
        # Check gradient
        assert x.grad is not None
        assert x.grad.shape == x.shape
        
        # Test bfloat16 if available
        bf16_supported = False
        if torch.cuda.is_bf16_supported():
            x_bf16 = torch.randn(100, 100, device="cuda", dtype=torch.bfloat16)
            y_bf16 = torch.randn(100, 100, device="cuda", dtype=torch.bfloat16)
            z_bf16 = torch.mm(x_bf16, y_bf16)
            bf16_supported = True
        
        results["test_passed"] = True
        results["bf16_supported"] = bf16_supported
        
    except Exception as e:
        results["error"] = str(e)
    
    return results