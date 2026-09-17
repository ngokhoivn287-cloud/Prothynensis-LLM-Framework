"""Tests for strict real-data mode in dataset loaders."""

from __future__ import annotations

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.data.real_datasets import (
    BoundedImageNetLoader,
    BoundedComputerUseLoader,
    assert_real_dataset_sample,
)
from src.utils.paths import get_primary_root


def _cleanup_metadata():
    for name in ["imagenet_1k_mrtp", "computer_use_mrtp"]:
        p = get_primary_root() / "datasets" / name / "metadata.json"
        if p.exists():
            p.unlink()


def test_imagenet_strict_mode_rejects_fallback(monkeypatch):
    _cleanup_metadata()
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("HF auth required")),
    )
    loader = BoundedImageNetLoader(max_samples=2, split="train", synthetic_fallback=False)
    with pytest.raises(RuntimeError, match="REAL DATA UNAVAILABLE"):
        loader.prepare()


def test_computer_use_strict_mode_rejects_fallback(monkeypatch):
    _cleanup_metadata()
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("HF auth required")),
    )
    loader = BoundedComputerUseLoader(max_samples=2, max_frames=2, synthetic_fallback=False)
    with pytest.raises(RuntimeError, match="REAL DATA UNAVAILABLE"):
        loader.prepare()


def test_real_sample_provenance():
    from src.utils.paths import get_primary_root
    cache_root = get_primary_root() / "cache" / "imagenet_1k_mrtp"
    sample = {
        "sample_id": "imagenet_train_000001",
        "path": str(cache_root / "imagenet_train_000001.png"),
        "label": 42,
        "source": "ILSVRC/imagenet-1k",
        "is_synthetic": False,
    }
    assert_real_dataset_sample(sample)


def test_synthetic_sample_rejected():
    from src.utils.paths import get_primary_root
    cache_root = get_primary_root() / "cache" / "imagenet_1k_mrtp"
    sample = {
        "sample_id": "imagenet_train_syn_000001",
        "path": str(cache_root / "imagenet_train_syn_000001.png"),
        "source": "synthetic_fallback",
        "is_synthetic": True,
    }
    with pytest.raises(ValueError, match="Synthetic fallback sample"):
        assert_real_dataset_sample(sample)


def test_mrtp_real_mode_disables_synthetic_fallback():
    loader = BoundedImageNetLoader(max_samples=2, split="train", synthetic_fallback=False)
    assert loader.synthetic_fallback is False
    loader_cu = BoundedComputerUseLoader(max_samples=2, max_frames=2, synthetic_fallback=False)
    assert loader_cu.synthetic_fallback is False
