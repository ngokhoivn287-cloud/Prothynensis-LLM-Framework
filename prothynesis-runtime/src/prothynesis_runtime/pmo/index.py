from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ModelEntry:
    model_id: str
    path: str
    format: str = "safetensors"
    size_mb: float = 0.0
    checksum: str = ""
    loaded: bool = False
    vram_used_gb: float = 0.0


class ModelIndex:
    def __init__(self) -> None:
        self.models: Dict[str, ModelEntry] = {}

    def register(self, entry: ModelEntry) -> None:
        self.models[entry.model_id] = entry

    def unregister(self, model_id: str) -> None:
        self.models.pop(model_id, None)

    def get(self, model_id: str) -> Optional[ModelEntry]:
        return self.models.get(model_id)

    def list_loaded(self) -> list[str]:
        return [mid for mid, entry in self.models.items() if entry.loaded]

    def get_loaded_count(self) -> int:
        return sum(1 for entry in self.models.values() if entry.loaded)

    def track_vram(self, model_id: str, vram_gb: float) -> None:
        if model_id in self.models:
            self.models[model_id].vram_used_gb = vram_gb

    def load_model(self, model_id: str) -> Any:
        entry = self.models.get(model_id)
        if not entry:
            raise FileNotFoundError(f"Model '{model_id}' not registered")
        entry.loaded = True
        return None
