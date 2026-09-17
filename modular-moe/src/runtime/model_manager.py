"""Model management for Prothynesis Runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from momm.registry import ModelRegistry, create_registry, ModelMetadata
from momm.model import ModelConfig, create_model
from src.utils.config import load_config, save_config
from src.runtime.settings import SettingsManager


@dataclass
class ModelInfo:
    model_id: str
    name: str
    path: str
    format: str
    size_mb: float
    parameter_count: int = 0
    architecture: Dict[str, Any] = field(default_factory=dict)
    specialization: str = "general"
    status: str = "available"
    version: str = "1.0"
    checksum: str = ""


class RuntimeModelManager:
    """Manages model discovery, import, verification, and lifecycle."""
    
    def __init__(self, settings: SettingsManager):
        self.settings = settings
        self.registry = create_registry(
            registry_dir=settings.get("models.registry_path", "models"),
            max_gpu_models=settings.get("runtime.max_gpu_models", 2),
            max_cpu_models=settings.get("runtime.max_cpu_models", 8),
            lazy_loading=settings.get("runtime.lazy_loading", True),
            cpu_offload=settings.get("runtime.cpu_offload", True),
        )
    
    def discover_models(self, search_paths: Optional[List[Path]] = None) -> List[ModelInfo]:
        """Discover model files in local storage."""
        if search_paths is None:
            storage = Path(self.settings.get("models.storage_path", "models"))
            search_paths = [storage]
        
        discovered = []
        for base in search_paths:
            if not base.exists():
                continue
            for path in base.rglob("*"):
                if path.is_file() and path.suffix.lower() in {".pt", ".safetensors", ".gguf", ".pmo"}:
                    info = ModelInfo(
                        model_id=path.stem,
                        name=path.name,
                        path=str(path),
                        format=path.suffix.lower().lstrip("."),
                        size_mb=path.stat().st_size / (1024 * 1024),
                    )
                    discovered.append(info)
        return discovered
    
    def verify_model(self, model_path: Path) -> Dict[str, Any]:
        """Verify model integrity and return metadata."""
        result = {
            "valid": False,
            "path": str(model_path),
            "format": model_path.suffix.lower().lstrip("."),
            "errors": [],
            "metadata": {},
        }
        
        if not model_path.exists():
            result["errors"].append("File not found")
            return result
        
        try:
            if model_path.suffix.lower() == ".safetensors":
                from safetensors import safe_open
                with safe_open(model_path, framework="pt") as f:
                    keys = list(f.keys())
                    result["metadata"]["num_keys"] = len(keys)
                    result["metadata"]["sample_keys"] = keys[:10]
                    result["valid"] = True
            elif model_path.suffix.lower() == ".pt":
                state = torch.load(model_path, map_location="cpu")
                if isinstance(state, dict):
                    result["metadata"]["num_keys"] = len(state.keys())
                    result["metadata"]["sample_keys"] = list(state.keys())[:10]
                result["valid"] = True
            elif model_path.suffix.lower() == ".gguf":
                try:
                    import gguf
                    with open(model_path, "rb") as f:
                        reader = gguf.GGUFReader(f)
                        result["metadata"]["num_keys"] = len(reader.fields)
                        result["metadata"]["sample_keys"] = list(reader.fields.keys())[:10]
                    result["valid"] = True
                except Exception as e:
                    result["errors"].append(f"GGUF read error: {e}")
            elif model_path.suffix.lower() == ".pmo":
                from pmo.package import PMOPackage
                pmo = PMOPackage(str(model_path))
                result["valid"] = pmo.verify()
                result["metadata"]["inspect"] = pmo.inspect()
                if not result["valid"]:
                    result["errors"].append("PMO verification failed")
            else:
                result["errors"].append(f"Unsupported format: {model_path.suffix}")
        except Exception as e:
            result["errors"].append(str(e))
        
        return result
    
    def import_model(self, source_path: Path, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Import a local model into the registry."""
        verification = self.verify_model(source_path)
        if not verification["valid"]:
            return {"success": False, "errors": verification["errors"]}
        
        model_id = model_id or source_path.stem
        
        metadata = ModelMetadata(
            model_id=model_id,
            checkpoint_path=str(source_path),
            parameter_count=0,
            architecture={"format": source_path.suffix.lower().lstrip("."), **verification.get("metadata", {})},
            tokenizer_id="default",
            specialization="general",
            version="1.0",
            training_status="untrained",
            source="local_import",
        )
        
        self.registry.models[model_id] = metadata
        self.registry._save_manifest()
        
        return {"success": True, "model_id": model_id, "metadata": metadata.to_dict()}
    
    def register_trained_model(self, model: Any, specialization: str = "general") -> ModelMetadata:
        """Register a trained model in the registry."""
        model_id = getattr(model, "model_id", "unknown")
        checkpoint_path = Path(self.settings.get("models.registry_path", "models")) / f"{model_id}.pt"
        
        torch.save(model.state_dict(), checkpoint_path)
        
        metadata = ModelMetadata(
            model_id=model_id,
            checkpoint_path=str(checkpoint_path),
            parameter_count=model.get_parameter_count()["total"],
            architecture=model.get_config(),
            tokenizer_id="default",
            specialization=specialization,
            version="1.0",
            training_status="trained",
        )
        
        self.registry.register_model(
            model=model,
            checkpoint_path=str(checkpoint_path),
            specialization=specialization,
        )
        return metadata
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get info about a registered model."""
        if model_id not in self.registry.models:
            return None
        meta = self.registry.models[model_id]
        return meta.to_dict()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        return [m.to_dict() for m in self.registry.models.values()]
