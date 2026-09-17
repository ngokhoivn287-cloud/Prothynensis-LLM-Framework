"""Settings management for Prothynesis Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict, field

from src.utils.config import load_config, save_config, get_nested, set_nested


@dataclass
class HardwareProfile:
    gpu_name: str = ""
    gpu_count: int = 0
    vram_total_gb: float = 0.0
    vram_free_gb: float = 0.0
    cpu_memory_gb: float = 0.0
    cpu_available_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    platform: str = ""
    python_version: str = ""
    pytorch_version: str = ""
    compute_capability: Optional[str] = None


@dataclass
class RuntimePaths:
    app_data: Path = field(default_factory=lambda: Path.home() / ".local" / "Prothynesis")
    models: Path = field(default_factory=lambda: Path.home() / ".local" / "Prothynesis" / "Models")
    downloads: Path = field(default_factory=lambda: Path.home() / ".local" / "Prothynesis" / "Downloads")
    logs: Path = field(default_factory=lambda: Path.home() / ".local" / "Prothynesis" / "Logs")
    cache: Path = field(default_factory=lambda: Path.home() / ".local" / "Prothynesis" / "Cache")
    registry: Path = field(default_factory=lambda: Path("models"))
    
    def ensure_directories(self) -> None:
        for path in [self.app_data, self.models, self.downloads, self.logs, self.cache]:
            path.mkdir(parents=True, exist_ok=True)


class SettingsManager:
    """Manages runtime settings persistence and defaults."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".local" / "Prothynesis" / "settings.yaml"
        self.paths = RuntimePaths()
        self.paths.ensure_directories()
        self._data = self._load()
    
    def _load(self) -> Dict[str, Any]:
        if self.config_path.exists():
            try:
                return load_config(self.config_path)
            except Exception:
                pass
        return self._defaults()
    
    def _defaults(self) -> Dict[str, Any]:
        return {
            "app": {
                "name": "Prothynesis Runtime",
                "version": "0.2.0",
                "mode": "simple",
                "theme": "dark",
                "first_run": True,
            },
            "runtime": {
                "backend": "auto",
                "device": "auto",
                "max_gpu_models": 2,
                "max_cpu_models": 8,
                "lazy_loading": True,
                "cpu_offload": True,
                "compile_models": False,
                "use_flash_attn": False,
            },
            "generation": {
                "default_top_k": 2,
                "default_temperature": 1.0,
                "default_top_p": 1.0,
                "default_max_new_tokens": 100,
                "default_repetition_penalty": 1.0,
            },
            "models": {
                "storage_path": str(self.paths.models),
                "registry_path": str(self.paths.registry),
                "default_context_length": 2048,
                "default_quantization": "bf16",
            },
            "hardware": {},
            "ui": {
                "theme": "dark",
                "mode": "simple",
            },
            "privacy": {
                "telemetry_enabled": False,
                "offline_mode": False,
                "auto_download": True,
            },
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)
    
    def save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        save_config(self._data, self.config_path)
    
    def get(self, key: str, default: Any = None) -> Any:
        return get_nested(self._data, key, default)
    
    def set(self, key: str, value: Any) -> None:
        set_nested(self._data, key, value)
        self.save()
    
    def get_hardware_profile(self) -> HardwareProfile:
        """Build hardware profile from stored/system data."""
        from src.utils.hardware import get_system_info
        info = get_system_info()
        
        gpu_name = ""
        vram_total = 0.0
        vram_free = 0.0
        compute_cap = None
        if info.get("num_gpus", 0) > 0:
            gpu_key = f"gpu_0"
            gpu_info = info.get(gpu_key, {})
            gpu_name = gpu_info.get("name", "")
            vram_total = gpu_info.get("total_memory_gb", 0.0)
            vram_free = gpu_info.get("free_memory_gb", 0.0)
            compute_cap = gpu_info.get("compute_capability")
        
        return HardwareProfile(
            gpu_name=gpu_name,
            gpu_count=info.get("num_gpus", 0),
            vram_total_gb=vram_total,
            vram_free_gb=vram_free,
            cpu_memory_gb=info.get("cpu_memory_gb", 0.0),
            cpu_available_gb=info.get("cpu_available_gb", 0.0),
            cuda_available=info.get("cuda_available", False),
            cuda_version=info.get("cuda_version"),
            cudnn_version=info.get("cudnn_version"),
            platform=info.get("platform", ""),
            python_version=info.get("python_version", ""),
            pytorch_version=info.get("pytorch_version", ""),
            compute_capability=compute_cap,
        )
    
    def recommend_configuration(self) -> Dict[str, Any]:
        """Recommend runtime configuration based on hardware."""
        profile = self.get_hardware_profile()
        recommendations = {
            "backend": "cpu",
            "max_gpu_models": 0,
            "max_cpu_models": 8,
            "default_context_length": 2048,
            "quantization": "bf16",
            "notes": [],
        }
        
        if profile.gpu_count > 0 and profile.vram_total_gb > 0:
            recommendations["backend"] = "cuda"
            if profile.vram_total_gb >= 24:
                recommendations["max_gpu_models"] = 4
                recommendations["default_context_length"] = 4096
                recommendations["notes"].append("High-end GPU detected")
            elif profile.vram_total_gb >= 12:
                recommendations["max_gpu_models"] = 2
                recommendations["default_context_length"] = 2048
                recommendations["quantization"] = "bf16"
                recommendations["notes"].append("Mid-range GPU detected")
            else:
                recommendations["max_gpu_models"] = 1
                recommendations["default_context_length"] = 1024
                recommendations["quantization"] = "q4_k_m"
                recommendations["notes"].append("Low VRAM GPU detected - consider quantization")
        else:
            recommendations["backend"] = "cpu"
            recommendations["max_gpu_models"] = 0
            recommendations["max_cpu_models"] = max(1, profile.cpu_available_gb // 4)
            recommendations["default_context_length"] = 1024
            recommendations["quantization"] = "q4_k_m"
            recommendations["notes"].append("CPU-only mode - performance will be limited")
        
        return recommendations
    
    def apply_recommendations(self) -> None:
        """Apply hardware-based recommendations to settings."""
        recs = self.recommend_configuration()
        for key, value in recs.items():
            if key in ["backend", "max_gpu_models", "max_cpu_models", "default_context_length"]:
                self.set(f"runtime.{key}", value)
        self.set("models.default_quantization", recs.get("quantization", "bf16"))
