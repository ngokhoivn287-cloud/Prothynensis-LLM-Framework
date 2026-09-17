import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class RuntimePaths:
    config_dir: Path = field(default_factory=lambda: Path(os.getenv("LOCALAPPDATA", str(Path.home() / ".local"))) / "Prothynesis")
    settings_file: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    cache_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    workers_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.settings_file = self.config_dir / "settings.yaml"
        self.artifacts_dir = self.config_dir / "artifacts"
        self.models_dir = self.config_dir / "models"
        self.cache_dir = self.config_dir / "cache"
        self.logs_dir = self.config_dir / "logs"
        self.workers_dir = self.config_dir / "workers"


@dataclass
class HardwareProfile:
    backend: str = "cpu"
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    cpu_cores: int = 1
    ram_gb: float = 0.0
    max_gpu_models: int = 0
    max_cpu_models: int = 1
    context_length: int = 2048


@dataclass
class SettingsManager:
    app_name: str = "Prothynesis"
    version: str = "0.1.0"
    paths: RuntimePaths = field(default_factory=RuntimePaths)
    hardware: HardwareProfile = field(default_factory=HardwareProfile)

    runtime: Dict[str, Any] = field(default_factory=lambda: {
        "device": "auto",
        "dtype": "float32",
        "threads": 4,
        "runtime_version": "0.1.0",
    })

    generation: Dict[str, Any] = field(default_factory=lambda: {
        "default_max_tokens": 512,
        "default_temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repetition_penalty": 1.05,
        "streaming": True,
    })

    models: Dict[str, Any] = field(default_factory=lambda: {
        "default_model_id": None,
        "auto_download": False,
        "trust_remote_code": False,
        "lazy_loading": True,
        "cpu_offload": False,
        "max_gpu_models": 1,
        "max_cpu_models": 1,
    })

    hardware_detected: Dict[str, Any] = field(default_factory=dict)

    ui: Dict[str, Any] = field(default_factory=lambda: {
        "theme": "dark",
        "language": "en",
        "start_minimized": False,
        "show_system_tray": True,
    })

    privacy: Dict[str, Any] = field(default_factory=lambda: {
        "telemetry_enabled": False,
        "crash_reporting": False,
        "auto_update": True,
    })

    worker: Dict[str, Any] = field(default_factory=lambda: {
        "worker_enabled": False,
        "max_concurrent_solvers": 1,
        "max_gpu_utilization": 0.9,
        "max_vram_usage_gb": None,
        "pause_while_user_active": False,
        "max_temperature_c": None,
    })

    github: Dict[str, Any] = field(default_factory=lambda: {
        "connected": False,
        "oauth_token": None,
        "username": None,
    })

    _raw: Dict[str, Any] = field(default_factory=dict, repr=False)

    def _ensure_dirs(self) -> None:
        self.paths.config_dir.mkdir(parents=True, exist_ok=True)
        self.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)
        self.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        self.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.paths.workers_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        obj: Any = self
        for part in parts:
            if isinstance(obj, dict):
                obj = obj.get(part, default)
            else:
                obj = getattr(obj, part, default)
        if obj is None:
            return default
        return obj

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        target = self._raw
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value

    def save(self) -> None:
        self._ensure_dirs()
        data = asdict(self)
        data.pop("paths", None)
        data.pop("_raw", None)
        with open(self.paths.settings_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def load(self) -> None:
        if not self.paths.settings_file.exists():
            self._raw = {}
            return
        with open(self.paths.settings_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        self._raw = loaded
        for key, value in loaded.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, dict) and isinstance(value, dict):
                    attr.clear()
                    attr.update(value)
                else:
                    setattr(self, key, value)

    def recommend_configuration(self) -> Dict[str, Any]:
        recommendations: Dict[str, Any] = {}
        recommendations["hardware"] = {
            "backend": self.hardware.backend,
            "max_gpu_models": self.hardware.max_gpu_models,
            "max_cpu_models": self.hardware.max_cpu_models,
            "context_length": self.hardware.context_length,
        }
        recommendations["models"] = {
            "max_gpu_models": self.hardware.max_gpu_models,
            "max_cpu_models": self.hardware.max_cpu_models,
            "lazy_loading": True,
            "cpu_offload": self.hardware.backend != "cuda",
        }
        recommendations["worker"] = {
            "worker_enabled": False,
            "max_concurrent_solvers": min(self.hardware.max_gpu_models, 1),
            "max_gpu_utilization": 0.9 if self.hardware.backend == "cuda" else 0.0,
        }
        return recommendations

    def apply_recommendations(self) -> None:
        recommendations = self.recommendation = self.recommend_configuration()
        if "hardware" in recommendations:
            hw = recommendations["hardware"]
            self.hardware.backend = hw.get("backend", self.hardware.backend)
            self.hardware.max_gpu_models = hw.get("max_gpu_models", self.hardware.max_gpu_models)
            self.hardware.max_cpu_models = hw.get("max_cpu_models", self.hardware.max_cpu_models)
            self.hardware.context_length = hw.get("context_length", self.hardware.context_length)
        if "models" in recommendations:
            m = recommendations["models"]
            self.models["max_gpu_models"] = m.get("max_gpu_models", self.models.get("max_gpu_models", 1))
            self.models["max_cpu_models"] = m.get("max_cpu_models", self.models.get("max_cpu_models", 1))
            self.models["lazy_loading"] = m.get("lazy_loading", self.models.get("lazy_loading", True))
            self.models["cpu_offload"] = m.get("cpu_offload", self.models.get("cpu_offload", False))
        if "worker" in recommendations:
            w = recommendations["worker"]
            self.worker.update(w)
        self.save()
