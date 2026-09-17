"""Execution environment detection and policy for Prothynensis.

Policy:
1. WSL2 + NVIDIA CUDA (preferred)
2. Windows + NVIDIA CUDA
3. CPU fallback only when explicitly requested or GPU validation unavailable
"""

from __future__ import annotations

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch

logger = logging.getLogger(__name__)


class ExecutionEnvironment:
    UNKNOWN = "unknown"
    WSL2_CUDA = "wsl2_cuda"
    WINDOWS_CUDA = "windows_cuda"
    CPU = "cpu"


class EnvironmentDetector:
    """Detect available execution environment."""

    @staticmethod
    def detect_wsl2() -> bool:
        """Check if running in WSL2."""
        if sys.platform != "linux":
            return False
        try:
            with open("/proc/version", "r") as f:
                content = f.read().lower()
            return "microsoft" in content or "wsl" in content
        except Exception:
            return False

    @staticmethod
    def detect_windows_cuda() -> bool:
        """Check if Windows CUDA is available."""
        if sys.platform != "win32":
            return False
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def detect_cuda_in_wsl2() -> bool:
        """Check if CUDA is available in WSL2."""
        if not EnvironmentDetector.detect_wsl2():
            return False
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except Exception:
            return False

    @staticmethod
    def detect_environment() -> str:
        """Detect the best available execution environment."""
        if EnvironmentDetector.detect_cuda_in_wsl2():
            return ExecutionEnvironment.WSL2_CUDA
        if EnvironmentDetector.detect_windows_cuda():
            return ExecutionEnvironment.WINDOWS_CUDA
        return ExecutionEnvironment.CPU

    @staticmethod
    def get_environment_info() -> Dict[str, Any]:
        """Get detailed environment information."""
        info = {
            "environment": EnvironmentDetector.detect_environment(),
            "os": sys.platform,
            "python_executable": sys.executable,
            "pytorch_version": None,
            "pytorch_cuda_build": None,
            "cuda_runtime": None,
            "nvidia_driver": None,
            "gpu_name": None,
            "cuda_available": False,
            "device_selected": None,
        }

        try:
            import torch
            info["pytorch_version"] = torch.__version__
            info["pytorch_cuda_build"] = torch.version.cuda
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["device_selected"] = "cuda"
            else:
                info["device_selected"] = "cpu"
        except Exception as e:
            logger.warning(f"Failed to get PyTorch info: {e}")
            info["device_selected"] = "cpu"

        if EnvironmentDetector.detect_windows_cuda():
            try:
                import torch
                if torch.cuda.is_available():
                    info["nvidia_driver"] = torch.cuda.get_device_properties(0).driver_version if hasattr(torch.cuda.get_device_properties(0), 'driver_version') else "unknown"
            except Exception:
                pass

        return info

    @staticmethod
    def get_device(force_cpu: bool = False) -> torch.device:
        """Get the best available device."""
        if force_cpu:
            logger.info("CPU mode forced by configuration")
            return torch.device("cpu")

        env = EnvironmentDetector.detect_environment()
        if env == ExecutionEnvironment.WSL2_CUDA:
            logger.info("Using WSL2 + CUDA")
            return torch.device("cuda")
        elif env == ExecutionEnvironment.WINDOWS_CUDA:
            logger.info("Using Windows + CUDA")
            return torch.device("cuda")
        else:
            logger.warning("No CUDA available, falling back to CPU")
            return torch.device("cpu")

    @staticmethod
    def report_environment() -> str:
        """Generate a human-readable environment report."""
        info = EnvironmentDetector.get_environment_info()
        lines = [
            "Environment:",
            f"  OS: {info['os']}",
            f"  Python: {info['python_executable']}",
            f"  PyTorch: {info['pytorch_version']}",
            f"  PyTorch CUDA build: {info['pytorch_cuda_build']}",
            f"  CUDA runtime: {info['cuda_runtime'] or 'N/A'}",
            f"  NVIDIA driver: {info['nvidia_driver'] or 'N/A'}",
            f"  GPU: {info['gpu_name'] or 'N/A'}",
            f"  torch.cuda.is_available(): {info['cuda_available']}",
            f"  Device selected: {info['device_selected']}",
        ]
        return "\n".join(lines)
