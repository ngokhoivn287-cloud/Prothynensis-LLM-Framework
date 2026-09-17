"""Model Registry for MoMMs - dynamic model discovery and management."""

from __future__ import annotations

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading

import torch

from .model import Model, ModelConfig, create_model


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_id: str
    checkpoint_path: str
    parameter_count: int
    architecture: Dict[str, Any]
    tokenizer_id: str
    specialization: str
    version: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: Dict[str, float] = field(default_factory=dict)
    checksum: str = ""
    training_status: str = "untrained"  # untrained, training, trained, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelMetadata":
        return cls(**d)


class ModelRegistry:
    """
    Model Registry for MoMMs.
    
    Manages model discovery, loading, unloading, and metadata.
    Supports dynamic model loading with lazy loading and CPU offloading.
    """
    
    def __init__(
        self,
        registry_dir: str = "models",
        max_gpu_models: int = 2,
        max_cpu_models: int = 8,
        lazy_loading: bool = True,
        cpu_offload: bool = True,
    ):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_gpu_models = max_gpu_models
        self.max_cpu_models = max_cpu_models
        self.lazy_loading = lazy_loading
        self.cpu_offload = cpu_offload
        
        # Model metadata
        self.manifest_path = self.registry_dir / "models_manifest.json"
        self.models: Dict[str, ModelMetadata] = {}
        
        # Loaded models cache
        self._gpu_models: Dict[str, Model] = {}
        self._cpu_models: Dict[str, Model] = {}
        
        # LRU tracking
        self._gpu_access_order: List[str] = []
        self._cpu_access_order: List[str] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load manifest
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load model manifest from disk."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r") as f:
                    data = json.load(f)
                for m in data.get("models", []):
                    metadata = ModelMetadata.from_dict(m)
                    self.models[metadata.model_id] = metadata
            except Exception as e:
                print(f"Warning: Could not load manifest: {e}")
    
    def _save_manifest(self) -> None:
        """Save model manifest to disk."""
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "models": [m.to_dict() for m in self.models.values()],
        }
        with open(self.manifest_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        model: Model,
        checkpoint_path: str,
        specialization: str = "general",
        tokenizer_id: str = "default",
        version: str = "1.0",
        metrics: Optional[Dict[str, float]] = None,
    ) -> ModelMetadata:
        """Register a trained model."""
        with self._lock:
            model_id = model.model_id
            
            # Compute checksum
            checkpoint_path = Path(checkpoint_path)
            checksum = self._compute_checksum(checkpoint_path) if checkpoint_path.exists() else ""
            
            metadata = ModelMetadata(
                model_id=model_id,
                checkpoint_path=str(checkpoint_path),
                parameter_count=model.get_parameter_count()["total"],
                architecture=model.get_config(),
                tokenizer_id=tokenizer_id,
                specialization=specialization,
                version=version,
                metrics=metrics or {},
                checksum=checksum,
                training_status="trained",
            )
            
            self.models[model_id] = metadata
            self._save_manifest()
            
            return metadata
    
    def _compute_checksum(self, path: Path) -> str:
        """Compute SHA256 checksum of checkpoint file."""
        if not path.exists() or not path.is_file():
            return ""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model."""
        with self._lock:
            if model_id not in self.models:
                return False
            
            # Unload if loaded
            self.unload_model(model_id)
            
            del self.models[model_id]
            self._save_manifest()
            return True
    
    def load_model(self, model_id: str, device: Optional[torch.device] = None) -> Model:
        """Load a model into memory (GPU or CPU)."""
        with self._lock:
            if model_id not in self.models:
                raise ValueError(f"Model {model_id} not registered")
            
            # Check if already loaded
            if model_id in self._gpu_models:
                self._touch_gpu(model_id)
                return self._gpu_models[model_id]
            
            if model_id in self._cpu_models:
                self._touch_cpu(model_id)
                return self._cpu_models[model_id]
            
            # Load from checkpoint
            metadata = self.models[model_id]
            checkpoint_path = Path(metadata.checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Create model from metadata
            config_dict = metadata.architecture
            config = ModelConfig.from_dict(config_dict)
            model = create_model(config)
            
            # Load weights
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            
            # Determine target device
            target_device = device or self.device
            
            if target_device.type == "cuda":
                return self._load_to_gpu(model_id, model)
            else:
                return self._load_to_cpu(model_id, model)
    
    def _load_to_gpu(self, model_id: str, model: Model) -> Model:
        """Load model to GPU with LRU eviction."""
        # Evict if at capacity
        while len(self._gpu_models) >= self.max_gpu_models:
            self._evict_lru_gpu()
        
        model = model.to(self.device)
        self._gpu_models[model_id] = model
        self._gpu_access_order.append(model_id)
        return model
    
    def _load_to_cpu(self, model_id: str, model: Model) -> Model:
        """Load model to CPU with LRU eviction."""
        # Evict if at capacity
        while len(self._cpu_models) >= self.max_cpu_models:
            self._evict_lru_cpu()
        
        model = model.to("cpu")
        self._cpu_models[model_id] = model
        self._cpu_access_order.append(model_id)
        return model
    
    def _evict_lru_gpu(self) -> None:
        """Evict least recently used model from GPU."""
        if not self._gpu_access_order:
            return
        
        lru_id = self._gpu_access_order.pop(0)
        if lru_id in self._gpu_models:
            model = self._gpu_models.pop(lru_id)
            # Optionally move to CPU if CPU offload enabled
            if self.cpu_offload and len(self._cpu_models) < self.max_cpu_models:
                self._cpu_models[lru_id] = model.cpu()
                self._cpu_access_order.append(lru_id)
            else:
                del model
    
    def _evict_lru_cpu(self) -> None:
        """Evict least recently used model from CPU."""
        if not self._cpu_access_order:
            return
        
        lru_id = self._cpu_access_order.pop(0)
        if lru_id in self._cpu_models:
            del self._cpu_models[lru_id]
    
    def _touch_gpu(self, model_id: str) -> None:
        """Update LRU order for GPU model."""
        if model_id in self._gpu_access_order:
            self._gpu_access_order.remove(model_id)
        self._gpu_access_order.append(model_id)
    
    def _touch_cpu(self, model_id: str) -> None:
        """Update LRU order for CPU model."""
        if model_id in self._cpu_access_order:
            self._cpu_access_order.remove(model_id)
        self._cpu_access_order.append(model_id)
    
    def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory."""
        with self._lock:
            unloaded = False
            
            if model_id in self._gpu_models:
                del self._gpu_models[model_id]
                if model_id in self._gpu_access_order:
                    self._gpu_access_order.remove(model_id)
                unloaded = True
            
            if model_id in self._cpu_models:
                del self._cpu_models[model_id]
                if model_id in self._cpu_access_order:
                    self._cpu_access_order.remove(model_id)
                unloaded = True
            
            return unloaded
    
    def get_model(self, model_id: str) -> Optional[Model]:
        """Get a loaded model without loading if not present."""
        with self._lock:
            if model_id in self._gpu_models:
                self._touch_gpu(model_id)
                return self._gpu_models[model_id]
            if model_id in self._cpu_models:
                self._touch_cpu(model_id)
                return self._cpu_models[model_id]
            return None
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded in memory."""
        with self._lock:
            return model_id in self._gpu_models or model_id in self._cpu_models
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata."""
        with self._lock:
            return self.models.get(model_id)
    
    def list_models(self) -> List[ModelMetadata]:
        """List all registered models."""
        with self._lock:
            return list(self.models.values())
    
    def get_model_ids(self) -> List[str]:
        """Get all registered model IDs."""
        with self._lock:
            return list(self.models.keys())
    
    def get_loaded_models(self) -> Dict[str, str]:
        """Get currently loaded models and their locations."""
        with self._lock:
            result = {}
            for mid in self._gpu_models:
                result[mid] = "gpu"
            for mid in self._cpu_models:
                result[mid] = "cpu"
            return result
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        with self._lock:
            gpu_mem = 0
            cpu_mem = 0
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
            
            return {
                "gpu_models": len(self._gpu_models),
                "cpu_models": len(self._cpu_models),
                "gpu_memory_gb": gpu_mem,
                "max_gpu_models": self.max_gpu_models,
                "max_cpu_models": self.max_cpu_models,
                "loaded_on_gpu": list(self._gpu_models.keys()),
                "loaded_on_cpu": list(self._cpu_models.keys()),
            }
    
    def update_metadata(self, model_id: str, **kwargs) -> bool:
        """Update model metadata."""
        with self._lock:
            if model_id not in self.models:
                return False
            
            metadata = self.models[model_id]
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)
            
            metadata.updated_at = datetime.now().isoformat()
            self._save_manifest()
            return True
    
    def set_training_status(self, model_id: str, status: str) -> bool:
        """Update training status."""
        return self.update_metadata(model_id, training_status=status)
    
    def add_metrics(self, model_id: str, metrics: Dict[str, float]) -> bool:
        """Add metrics to model metadata."""
        with self._lock:
            if model_id not in self.models:
                return False
            
            metadata = self.models[model_id]
            metadata.metrics.update(metrics)
            metadata.updated_at = datetime.now().isoformat()
            self._save_manifest()
            return True
    
    def verify_checksum(self, model_id: str) -> bool:
        """Verify model checkpoint checksum."""
        with self._lock:
            if model_id not in self.models:
                return False
            
            metadata = self.models[model_id]
            path = Path(metadata.checkpoint_path)
            
            if not path.exists():
                return False
            
            actual = self._compute_checksum(path)
            return actual == metadata.checksum


def create_registry(
    registry_dir: str = "models",
    max_gpu_models: int = 2,
    max_cpu_models: int = 8,
    lazy_loading: bool = True,
    cpu_offload: bool = True,
) -> ModelRegistry:
    """Factory function to create a model registry."""
    return ModelRegistry(
        registry_dir=registry_dir,
        max_gpu_models=max_gpu_models,
        max_cpu_models=max_cpu_models,
        lazy_loading=lazy_loading,
        cpu_offload=cpu_offload,
    )