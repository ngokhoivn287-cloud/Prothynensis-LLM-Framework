"""Hardware detection for Prothynesis Runtime."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

import torch


@dataclass
class GPUInfo:
    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float
    compute_capability: Optional[str]
    driver_version: Optional[str]
    cuda_version: Optional[str]


@dataclass
class CPUInfo:
    model: str
    cores: int
    threads: int
    memory_gb: float
    memory_available_gb: float


@dataclass
class HardwareReport:
    platform: str
    python_version: str
    pytorch_version: str
    cuda_available: bool
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    cpu: CPUInfo
    gpus: List[GPUInfo]
    recommended_backend: str
    recommended_context: int
    recommended_max_gpu_models: int
    recommended_max_cpu_models: int
    warnings: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform,
            "python_version": self.python_version,
            "pytorch_version": self.pytorch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "cpu": asdict(self.cpu),
            "gpus": [asdict(g) for g in self.gpus],
            "recommended_backend": self.recommended_backend,
            "recommended_context": self.recommended_context,
            "recommended_max_gpu_models": self.recommended_max_gpu_models,
            "recommended_max_cpu_models": self.recommended_max_cpu_models,
            "warnings": self.warnings,
        }


def detect_hardware() -> HardwareReport:
    """Detect system hardware and recommend configuration."""
    import psutil
    
    # Platform info
    platform_info = platform.platform()
    python_version = platform.python_version()
    pytorch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else None
    cudnn_version = torch.backends.cudnn.version() if cuda_available else None
    
    # CPU info
    cpu_model = platform.processor() or platform.machine()
    cpu_cores = psutil.cpu_count(logical=False) or 1
    cpu_threads = psutil.cpu_count(logical=True) or 1
    mem = psutil.virtual_memory()
    cpu_memory_gb = mem.total / (1024**3)
    cpu_available_gb = mem.available / (1024**3)
    
    cpu_info = CPUInfo(
        model=cpu_model,
        cores=cpu_cores,
        threads=cpu_threads,
        memory_gb=cpu_memory_gb,
        memory_available_gb=cpu_available_gb,
    )
    
    # GPU info
    gpus: List[GPUInfo] = []
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = _get_gpu_memory(i)
            gpus.append(GPUInfo(
                index=i,
                name=props.name,
                vram_total_gb=total / (1024**3),
                vram_free_gb=free / (1024**3),
                compute_capability=f"{props.major}.{props.minor}",
                driver_version=_get_nvidia_driver_version(),
                cuda_version=cuda_version,
            ))
    
    # Recommendations
    warnings: List[str] = []
    if not cuda_available:
        warnings.append("CUDA not available - falling back to CPU inference")
    
    if gpus:
        total_vram = sum(g.vram_total_gb for g in gpus)
        if total_vram >= 24:
            recommended_backend = "cuda"
            recommended_context = 4096
            recommended_max_gpu_models = 4
            recommended_max_cpu_models = 4
        elif total_vram >= 12:
            recommended_backend = "cuda"
            recommended_context = 2048
            recommended_max_gpu_models = 2
            recommended_max_cpu_models = 4
        else:
            recommended_backend = "cuda"
            recommended_context = 1024
            recommended_max_gpu_models = 1
            recommended_max_cpu_models = 2
            warnings.append("Low VRAM GPU detected - consider smaller models or quantization")
    else:
        recommended_backend = "cpu"
        recommended_context = 1024
        recommended_max_gpu_models = 0
        recommended_max_cpu_models = max(1, int(cpu_available_gb // 4))
        warnings.append("CPU-only mode - inference will be slower")
    
    if cpu_available_gb < 4:
        warnings.append("Low system memory - performance may be limited")
    
    return HardwareReport(
        platform=platform_info,
        python_version=python_version,
        pytorch_version=pytorch_version,
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        cudnn_version=cudnn_version,
        cpu=cpu_info,
        gpus=gpus,
        recommended_backend=recommended_backend,
        recommended_context=recommended_context,
        recommended_max_gpu_models=recommended_max_gpu_models,
        recommended_max_cpu_models=recommended_max_cpu_models,
        warnings=warnings,
    )


def _get_gpu_memory(device_id: int) -> tuple[int, int]:
    """Get free and total GPU memory in bytes."""
    if not torch.cuda.is_available():
        return 0, 0
    
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        free, total = map(int, result.stdout.strip().split(","))
        return free * 1024**2, total * 1024**2
    except Exception:
        free, total = torch.cuda.mem_get_info(device_id)
        return free, total


def _get_nvidia_driver_version() -> Optional[str]:
    """Get NVIDIA driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return None
