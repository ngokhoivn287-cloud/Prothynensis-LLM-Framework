"""Bounded real-dataset loaders for Ultra 3 MrTP multimodal validation.

ALL operations resolve to:
    D:\\NgoPROJECT\\ProthynensisDatasets
    D:\\NgoPROJECT\\ProthynensisDatasets\\temp
"""

from __future__ import annotations

import io
import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from src.utils.paths import (
    is_allowed_dataset_path,
    assert_allowed_path,
    get_primary_root,
    get_temp_root,
    get_shards_root,
)

logger = logging.getLogger(__name__)


class BoundedDatasetError(Exception):
    """Raised when a bounded dataset operation would exceed limits."""
    pass


def _assert_approved(path: Any, context: str = "") -> Path:
    p = Path(path).resolve()
    if not is_allowed_dataset_path(p):
        raise PermissionError(f"Refusing path outside approved roots: {p} ({context})")
    return p


def assert_real_dataset_sample(sample: Dict[str, Any]) -> None:
    """Assert that a dataset sample is real, not synthetic fallback."""
    if not isinstance(sample, dict):
        raise TypeError(f"Sample must be a dict, got {type(sample)}")
    if "source" not in sample:
        raise ValueError("Sample missing required 'source' field")
    if sample.get("source") == "synthetic_fallback":
        raise ValueError("Synthetic fallback sample is not allowed in real-data validation mode")
    if sample.get("is_synthetic") is True:
        raise ValueError("Synthetic sample is not allowed in real-data validation mode")
    if "sample_id" not in sample:
        raise ValueError("Sample missing required 'sample_id' field")
    if "path" not in sample:
        raise ValueError("Sample missing required 'path' field")


class BoundedImageNetLoader:
    """Loads a tiny bounded subset of ImageNet-1K for MrTP validation.

    Does NOT download the full dataset.
    Uses streaming + bounded cache.
    """

    def __init__(self, max_samples: int = 64, split: str = "train", synthetic_fallback: bool = True):
        self.max_samples = max_samples
        self.split = split
        self.synthetic_fallback = synthetic_fallback
        self.root = get_primary_root() / "datasets" / "imagenet_1k_mrtp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_root = get_primary_root() / "cache" / "imagenet_1k_mrtp"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self._samples: List[Dict[str, Any]] = []

    def prepare(self, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Prepare bounded ImageNet subset.

        Returns list of sample dicts with:
            sample_id, path, label, class_id, domain, modality, split, source, is_synthetic
        """
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                data = json.load(f)
            self._samples = data.get("samples", [])
            if self._samples:
                return self._samples[: self.max_samples]

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for ImageNet loading")

        hf_id = "ILSVRC/imagenet-1k"
        subset = "default"

        logger.info(f"Loading bounded ImageNet subset: {hf_id} ({self.split}, max={self.max_samples})")
        try:
            ds = load_dataset(hf_id, subset, split=self.split, streaming=True)
        except Exception as e:
            if not self.synthetic_fallback:
                raise RuntimeError(
                    f"REAL DATA UNAVAILABLE: Failed to load ImageNet dataset '{hf_id}' from Hugging Face. "
                    f"Error: {e}. Hugging Face authentication may be required for this gated dataset."
                ) from e
            logger.warning(f"Failed to load ImageNet from HF: {e}")
            ds = None

        samples = []
        if ds is not None:
            for idx, item in enumerate(ds):
                if idx >= self.max_samples:
                    break
                image = item.get("image")
                label = item.get("label", -1)
                if image is None:
                    continue
                sample_id = f"imagenet_{self.split}_{idx:06d}"
                img_path = self.cache_root / f"{sample_id}.png"
                if not img_path.exists():
                    try:
                        image.save(img_path)
                    except Exception as e:
                        logger.debug(f"Skip sample {idx}: {e}")
                        continue
                samples.append({
                    "sample_id": sample_id,
                    "path": str(img_path),
                    "label": int(label) if label is not None else -1,
                    "class_id": int(label) if label is not None else -1,
                    "domain": "computer_vision",
                    "modality": "image",
                    "task": "image_classification",
                    "split": self.split,
                    "source": hf_id,
                    "is_synthetic": False,
                })
        elif self.synthetic_fallback:
            logger.warning("ImageNet streaming unavailable; creating synthetic image samples")
            samples = self._create_synthetic_image_samples()
        else:
            raise RuntimeError(
                "REAL DATA UNAVAILABLE: ImageNet dataset stream is None and synthetic_fallback is disabled. "
                "Cannot proceed with real-data validation."
            )

        if not samples:
            raise RuntimeError(
                "REAL DATA UNAVAILABLE: ImageNet loader returned zero samples. "
                "Check dataset access and streaming configuration."
            )

        metadata = {
            "dataset": hf_id,
            "subset": subset,
            "split": self.split,
            "max_samples": self.max_samples,
            "actual_count": len(samples),
            "synthetic_fallback": self.synthetic_fallback,
            "samples": samples,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._samples = samples
        return samples

    def _create_synthetic_image_samples(self) -> List[Dict[str, Any]]:
        """Create small synthetic image samples when ImageNet is unavailable."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("PIL required for synthetic image creation")

        samples = []
        for idx in range(min(self.max_samples, 32)):
            sample_id = f"imagenet_{self.split}_syn_{idx:06d}"
            img_path = self.cache_root / f"{sample_id}.png"
            if not img_path.exists():
                img = Image.new("RGB", (64, 64), color=(idx % 255, (idx * 2) % 255, (idx * 3) % 255))
                img.save(img_path)
            samples.append({
                "sample_id": sample_id,
                "path": str(img_path),
                "label": idx % 1000,
                "class_id": idx % 1000,
                "domain": "computer_vision",
                "modality": "image",
                "task": "image_classification",
                "split": self.split,
                "source": "synthetic_fallback",
            })
        return samples

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self.load()
        return self._samples

    def load(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self.prepare()
        return self._samples


class BoundedComputerUseLoader:
    """Loads a tiny bounded subset of computer-use data for MrTP validation.

    Uses streaming.
    Does NOT download the full dataset.
    """

    def __init__(self, max_samples: int = 32, max_frames: int = 4, synthetic_fallback: bool = True):
        self.max_samples = max_samples
        self.max_frames = max_frames
        self.synthetic_fallback = synthetic_fallback
        self.root = get_primary_root() / "datasets" / "computer_use_mrtp"
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache_root = get_primary_root() / "cache" / "computer_use_mrtp"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self._samples: List[Dict[str, Any]] = []

    def prepare(self, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Prepare bounded computer-use subset.

        Returns list of sample dicts with:
            sample_id, path, frames, actions, domain, modality, task, source, is_synthetic
        """
        if self.metadata_path.exists():
            with open(self.metadata_path, "r") as f:
                data = json.load(f)
            self._samples = data.get("samples", [])
            if self._samples:
                return self._samples[: self.max_samples]

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library required for computer-use loading")

        hf_id = "markov-ai/computer-use-large"

        logger.info(f"Loading bounded computer-use subset: {hf_id} (max={self.max_samples})")
        samples = []
        available = ["autocad", "blender", "excel", "photoshop", "salesforce", "vscode"]
        subset = "default" if "default" in available else available[0]
        try:
            ds = load_dataset(hf_id, subset, split="train", streaming=True)
        except Exception as e:
            if not self.synthetic_fallback:
                raise RuntimeError(
                    f"REAL DATA UNAVAILABLE: Failed to load computer-use dataset '{hf_id}' from Hugging Face. "
                    f"Error: {e}. Dataset may be unavailable or authentication may be required."
                ) from e
            logger.warning(f"Failed to load computer-use dataset with config={subset}: {e}")
            ds = None
        if ds is not None:
            import threading
            import time
            result_holder = {}
            def load_with_timeout():
                try:
                    local_samples = []
                    for idx, item in enumerate(ds):
                        if idx >= self.max_samples:
                            break
                        frames = item.get("frames") or item.get("images") or item.get("video")
                        actions = item.get("actions") or item.get("trajectory") or item.get("annotations")
                        sample_id = f"computer_use_{idx:06d}"
                        sample_dir = self.cache_root / sample_id
                        sample_dir.mkdir(parents=True, exist_ok=True)
                        frame_paths = []
                        if frames is not None:
                            try:
                                import numpy as np
                                from PIL import Image
                                if hasattr(frames, "shape"):
                                    frame_arr = np.asarray(frames)
                                    if frame_arr.ndim == 3:
                                        frame_arr = frame_arr[None]
                                    n = min(frame_arr.shape[0], self.max_frames)
                                    for fi in range(n):
                                        img = Image.fromarray(frame_arr[fi])
                                        fp = sample_dir / f"frame_{fi:03d}.png"
                                        img.save(fp)
                                        frame_paths.append(str(fp))
                                elif hasattr(frames, "__len__") and len(frames) > 0:
                                    for fi, frame in enumerate(frames[: self.max_frames]):
                                        if hasattr(frame, "save"):
                                            fp = sample_dir / f"frame_{fi:03d}.png"
                                            frame.save(fp)
                                            frame_paths.append(str(fp))
                                        elif hasattr(frame, "numpy"):
                                            arr = frame.cpu().numpy()
                                            if arr.dtype != np.uint8:
                                                arr = (arr * 255).astype(np.uint8)
                                            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                                                arr = arr.transpose(1, 2, 0)
                                            img = Image.fromarray(arr)
                                            fp = sample_dir / f"frame_{fi:03d}.png"
                                            img.save(fp)
                                            frame_paths.append(str(fp))
                            except Exception as e:
                                logger.debug(f"Skip frames for sample {idx}: {e}")
                        if not frame_paths:
                            continue
                        local_samples.append({
                            "sample_id": sample_id,
                            "path": str(sample_dir),
                            "frame_paths": frame_paths,
                            "actions": actions if isinstance(actions, (dict, list)) else {},
                            "domain": "computer_use",
                            "modality": "video",
                            "task": "gui_interaction",
                            "split": "train",
                            "source": hf_id,
                            "is_synthetic": False,
                        })
                    result_holder["samples"] = local_samples
                except Exception as e:
                    result_holder["error"] = str(e)
            t = threading.Thread(target=load_with_timeout)
            t.start()
            t.join(timeout=300)
            if t.is_alive():
                if not self.synthetic_fallback:
                    raise RuntimeError(
                        "REAL DATA UNAVAILABLE: Computer-use dataset loading timed out after 120s "
                        "and synthetic_fallback is disabled."
                    )
                logger.warning("Computer-use dataset loading timed out; using synthetic fallback")
                samples = []
            elif "error" in result_holder:
                if not self.synthetic_fallback:
                    raise RuntimeError(
                        f"REAL DATA UNAVAILABLE: Computer-use dataset loading error: {result_holder['error']}"
                    ) from RuntimeError(result_holder['error'])
                logger.warning(f"Computer-use dataset loading error: {result_holder['error']}")
                samples = []
            else:
                samples = result_holder.get("samples", [])
        
        if not samples and self.synthetic_fallback:
            logger.warning("Computer-use streaming yielded no samples; creating synthetic screen samples")
            samples = self._create_synthetic_computer_use_samples()
        elif not samples and not self.synthetic_fallback:
            raise RuntimeError(
                "REAL DATA UNAVAILABLE: Computer-use streaming yielded no samples and "
                "synthetic_fallback is disabled."
            )

        metadata = {
            "dataset": hf_id,
            "subset": subset,
            "max_samples": self.max_samples,
            "max_frames": self.max_frames,
            "actual_count": len(samples),
            "synthetic_fallback": self.synthetic_fallback,
            "samples": samples,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self._samples = samples
        return samples

    def _create_synthetic_computer_use_samples(self) -> List[Dict[str, Any]]:
        """Create synthetic screen samples when dataset is unavailable."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise ImportError("PIL required for synthetic screen samples")

        samples = []
        for idx in range(min(self.max_samples, 16)):
            sample_id = f"computer_use_syn_{idx:06d}"
            sample_dir = self.cache_root / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            frame_paths = []
            for fi in range(min(self.max_frames, 2)):
                fp = sample_dir / f"frame_{fi:03d}.png"
                if not fp.exists():
                    img = Image.new("RGB", (64, 64), color=(100 + idx % 100, 100 + fi * 20, 150))
                    draw = ImageDraw.Draw(img)
                    draw.rectangle([10 + fi * 5, 10 + fi * 5, 50 - fi * 5, 50 - fi * 5], outline="white")
                    img.save(fp)
                frame_paths.append(str(fp))
            samples.append({
                "sample_id": sample_id,
                "path": str(sample_dir),
                "frame_paths": frame_paths,
                "actions": {
                    "action_type": idx % 3,
                    "action_params": [10 + idx, 20 + idx, 30 + idx, 40 + idx],
                    "confidence": 0.8,
                },
                "domain": "computer_use",
                "modality": "video",
                "task": "gui_interaction",
                "split": "train",
                "source": "synthetic_fallback",
            })
        return samples

    def get_samples(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self.prepare()
        return self._samples

    def load(self) -> List[Dict[str, Any]]:
        if not self._samples:
            self.prepare()
        return self._samples
