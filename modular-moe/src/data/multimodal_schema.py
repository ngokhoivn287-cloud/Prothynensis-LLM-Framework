"""Multimodal dataset schema and adapters for Prothynensis."""

from __future__ import annotations

import io
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from enum import Enum


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    SCREEN = "screen"
    MULTIMODAL = "multimodal"


@dataclass
class MultimodalSample:
    sample_id: str
    modality: Union[Modality, str]
    domain: str
    task: str
    difficulty: str
    quality: float
    language: str = "en"
    source: str = ""
    payload: Optional[bytes] = None
    payload_path: Optional[str] = None
    annotations: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    checksum: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.modality, str):
            try:
                self.modality = Modality(self.modality)
            except ValueError:
                self.modality = Modality.MULTIMODAL

    def compute_checksum(self) -> str:
        if self.payload is not None:
            return hashlib.sha256(self.payload).hexdigest()
        if self.payload_path and Path(self.payload_path).exists():
            return hashlib.sha256(Path(self.payload_path).read_bytes()).hexdigest()
        return hashlib.sha256(self.sample_id.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value if isinstance(self.modality, Modality) else self.modality
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultimodalSample":
        d = dict(d)
        d["modality"] = d.get("modality", "multimodal")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetProfile:
    name: str
    domain: str
    difficulty: str
    task_type: str
    language: str = "en"
    quality_score: float = 0.8
    size: int = 1000
    source: str = "synthetic"
    license: str = "mit"
    modality: Union[Modality, str] = Modality.TEXT
    split: str = "train"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.modality, str):
            try:
                self.modality = Modality(self.modality)
            except ValueError:
                self.modality = Modality.TEXT

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["modality"] = self.modality.value if isinstance(self.modality, Modality) else self.modality
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetProfile":
        d = dict(d)
        d["modality"] = d.get("modality", "text")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ModalityAdapter:
    """Base adapter for modality-aware preprocessing."""

    def __init__(self, modality: Modality):
        self.modality = modality

    def can_handle(self, sample: MultimodalSample) -> bool:
        return sample.modality == self.modality

    def preprocess(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class TextAdapter(ModalityAdapter):
    def __init__(self):
        super().__init__(Modality.TEXT)

    def preprocess(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        if sample.payload is not None and isinstance(sample.payload, bytes):
            try:
                text = sample.payload.decode("utf-8", errors="ignore")
                return {"text": text, "tokens": list(text.encode("utf-8", errors="ignore"))}
            except Exception:
                return None
        if sample.payload_path:
            try:
                text = Path(sample.payload_path).read_text(encoding="utf-8", errors="ignore")
                return {"text": text, "tokens": list(text.encode("utf-8", errors="ignore"))}
            except Exception:
                return None
        return None


class ImageAdapter(ModalityAdapter):
    def __init__(self):
        super().__init__(Modality.IMAGE)

    def preprocess(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        try:
            from PIL import Image
            import io as _io
            if sample.payload is not None:
                img = Image.open(_io.BytesIO(sample.payload))
            elif sample.payload_path:
                img = Image.open(sample.payload_path)
            else:
                return None
            return {"image": img, "width": img.width, "height": img.height, "mode": img.mode}
        except Exception:
            return None


class VideoAdapter(ModalityAdapter):
    def __init__(self):
        super().__init__(Modality.VIDEO)

    def preprocess(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        try:
            import av
            if sample.payload_path:
                container = av.open(sample.payload_path)
                stream = container.streams.video[0]
                return {
                    "container": container,
                    "stream": stream,
                    "frames": stream.frames,
                    "duration": float(stream.duration * stream.time_base),
                }
            return None
        except Exception:
            return None


class ScreenAdapter(ModalityAdapter):
    def __init__(self):
        super().__init__(Modality.SCREEN)

    def preprocess(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        try:
            from PIL import Image
            import io as _io
            if sample.payload is not None:
                img = Image.open(_io.BytesIO(sample.payload))
            elif sample.payload_path:
                img = Image.open(sample.payload_path)
            else:
                return None
            return {
                "image": img,
                "width": img.width,
                "height": img.height,
                "screen_state": sample.annotations.get("screen_state", "unknown"),
            }
        except Exception:
            return None


class MultimodalDatasetRegistry:
    """Registry for multimodal datasets with modality-aware adapters."""

    def __init__(self):
        self.profiles: Dict[str, DatasetProfile] = {}
        self.adapters = {
            Modality.TEXT: TextAdapter(),
            Modality.IMAGE: ImageAdapter(),
            Modality.VIDEO: VideoAdapter(),
            Modality.SCREEN: ScreenAdapter(),
            Modality.MULTIMODAL: TextAdapter(),
        }

    def register_profile(self, profile: DatasetProfile) -> None:
        self.profiles[profile.name] = profile

    def get_adapter(self, sample: MultimodalSample) -> ModalityAdapter:
        return self.adapters.get(sample.modality, TextAdapter())

    def preprocess_sample(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        adapter = self.get_adapter(sample)
        if adapter.can_handle(sample):
            return adapter.preprocess(sample)
        return None

    def list_by_modality(self, modality: Union[Modality, str]) -> List[DatasetProfile]:
        if isinstance(modality, str):
            modality = Modality(modality)
        return [p for p in self.profiles.values() if p.modality == modality]

    def to_dict(self) -> Dict[str, Any]:
        return {name: p.to_dict() for name, p in self.profiles.items()}

    def save(self, output_path: str) -> None:
        p = Path(output_path)
        if not p.parent.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
