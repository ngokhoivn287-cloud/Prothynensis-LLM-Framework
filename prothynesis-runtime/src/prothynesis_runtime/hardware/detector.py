from dataclasses import dataclass, field
from typing import Optional

import psutil
import torch


@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    driver_version: Optional[str] = None


@dataclass
class CPUInfo:
    cores: int
    threads: int
    ram_gb: float
    ram_available_gb: float


@dataclass
class HardwareReport:
    backend: str = "cpu"
    gpus: list = field(default_factory=list)
    cpu: Optional[CPUInfo] = None
    recommendations: dict = field(default_factory=dict)


def _get_nvidia_driver_version() -> Optional[str]:
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if lines:
                return lines[0]
    except Exception:
        pass
    return None


def get_gpu_memory() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        free = total - reserved
        return total, free
    except Exception:
        return 0.0, 0.0


def get_cpu_memory() -> tuple[float, float]:
    vm = psutil.virtual_memory()
    total = vm.total / (1024 ** 3)
    available = vm.available / (1024 ** 3)
    return total, available


def detect_hardware() -> HardwareReport:
    report = HardwareReport()
    cpu = psutil.cpu_count(logical=False)
    threads = psutil.cpu_count(logical=True) or cpu
    ram_total, ram_available = get_cpu_memory()
    report.cpu = CPUInfo(
        cores=cpu,
        threads=threads,
        ram_gb=round(ram_total, 2),
        ram_available_gb=round(ram_available, 2),
    )

    if torch.cuda.is_available():
        report.backend = "cuda"
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            total, free = get_gpu_memory()
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / (1024 ** 3)
            report.gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    memory_total_gb=round(total, 2),
                    memory_free_gb=round(free, 2),
                    driver_version=_get_nvidia_driver_version(),
                )
            )

        if report.gpus:
            primary = report.gpus[0]
            max_gpu_models = 1 if primary.memory_total_gb >= 8 else 0
            context_length = 4096 if primary.memory_total_gb >= 16 else 2048
        else:
            max_gpu_models = 0
            context_length = 2048
    else:
        report.backend = "cpu"
        max_gpu_models = 0
        context_length = 2048

    max_cpu_models = max(1, report.cpu.cores // 2)

    report.recommendations = {
        "backend": report.backend,
        "max_gpu_models": max_gpu_models,
        "max_cpu_models": max_cpu_models,
        "context_length": context_length,
    }
    return report
