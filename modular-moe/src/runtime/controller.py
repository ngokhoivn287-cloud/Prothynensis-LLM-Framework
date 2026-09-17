"""Prothynesis Runtime - core controller shared by GUI and CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List

from momm.registry import ModelRegistry, create_registry
from momm.inference import InferenceEngine, InferenceConfig
from momm.coordinator import ModelCoordinator, CoordinatorConfig
from momm.model import ModelConfig
from src.utils.hardware import get_system_info
from src.utils.config import load_config, save_config, merge_configs
from src.runtime.settings import SettingsManager
from src.runtime.hardware import detect_hardware, HardwareReport
from src.runtime.model_manager import RuntimeModelManager
from src.runtime.startup import FirstRunSetup, run_startup
from src.runtime.runtime_orchestrator import RuntimeInferenceOrchestrator, TaskContext


class ProthynesisRuntime:
    """Main runtime controller for Prothynesis Runtime application."""
    
    def __init__(self, settings_path: Optional[Path] = None):
        self.settings = SettingsManager(settings_path)
        self.registry = create_registry(
            registry_dir=self.settings.get("models.registry_path", "models"),
            max_gpu_models=self.settings.get("runtime.max_gpu_models", 2),
            max_cpu_models=self.settings.get("runtime.max_cpu_models", 8),
            lazy_loading=self.settings.get("runtime.lazy_loading", True),
            cpu_offload=self.settings.get("runtime.cpu_offload", True),
        )
        self.model_manager = RuntimeModelManager(self.settings)
        self.inference_config = InferenceConfig(
            max_gpu_models=self.settings.get("runtime.max_gpu_models", 2),
            max_cpu_models=self.settings.get("runtime.max_cpu_models", 8),
            lazy_loading=self.settings.get("runtime.lazy_loading", True),
            cpu_offload=self.settings.get("runtime.cpu_offload", True),
            compile_models=self.settings.get("runtime.compile_models", False),
            use_flash_attn=self.settings.get("runtime.use_flash_attn", False),
            default_top_k=self.settings.get("generation.default_top_k", 2),
            default_temperature=self.settings.get("generation.default_temperature", 1.0),
            default_top_p=self.settings.get("generation.default_top_p", 1.0),
            default_max_new_tokens=self.settings.get("generation.default_max_new_tokens", 100),
            default_repetition_penalty=self.settings.get("generation.default_repetition_penalty", 1.0),
        )
        self.inference_engine = InferenceEngine(
            config=self.inference_config,
            registry=self.registry,
        )
        self.runtime_orchestrator = RuntimeInferenceOrchestrator(
            config=self.settings.to_dict(),
            registry=self.registry,
            inference_engine=self.inference_engine,
        )
        self._coordinator: Optional[ModelCoordinator] = None
    
    def get_hardware_info(self) -> Dict[str, Any]:
        """Get hardware detection results."""
        return detect_hardware().to_dict()
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime status info."""
        return {
            "version": self._get_version(),
            "settings": self.settings.to_dict(),
            "hardware": self.get_hardware_info(),
            "registry": {
                "total_models": len(self.registry.get_model_ids()),
                "loaded_models": len(self.registry.get_loaded_models()),
            },
        }
    
    def _get_version(self) -> str:
        try:
            from momm import __version__ as momm_version
            return momm_version
        except ImportError:
            return "0.2.0"

    def execute_with_dynamic_recruitment(
        self,
        task_context: TaskContext,
        available_models: List[Any],
        capability_requirements: Optional[Dict[str, float]] = None,
        verification_fn: Optional[Callable[[str], Any]] = None,
        synthesis_fn: Optional[Callable[[List[Any]], str]] = None,
    ) -> Dict[str, Any]:
        """Execute a task using dynamic runtime recruitment."""
        return self.runtime_orchestrator.execute_task(
            task_context=task_context,
            available_models=available_models,
            capability_requirements=capability_requirements,
            verification_fn=verification_fn,
            synthesis_fn=synthesis_fn,
        )
    
    def shutdown(self) -> None:
        """Clean shutdown."""
        try:
            if self._coordinator is not None:
                del self._coordinator
            self.registry.unload_all()
        except Exception:
            pass
