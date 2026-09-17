"""Tests for mini3 pilot training script."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

pytest.importorskip("torch")


class TestMini3Pilot:
    def test_script_imports(self):
        """Test that the mini3 pilot script can be imported."""
        import importlib.util
        script_path = Path(__file__).parent.parent / "scripts" / "train_mini3_pilot.py"
        spec = importlib.util.spec_from_file_location("train_mini3_pilot", str(script_path))
        module = importlib.util.module_from_spec(spec)
        sys.modules["train_mini3_pilot"] = module
        spec.loader.exec_module(module)
        assert hasattr(module, "run_pilot")
        assert hasattr(module, "build_parser")
        del sys.modules["train_mini3_pilot"]
    
    def test_dataset_uniqueness(self):
        """Test that unique shards have no duplicate samples."""
        from scripts.train_mini3_pilot import _build_synthetic_dataset, SharedUniqueDataset
        from src.data.tokenizer import CharacterTokenizer
        
        tokenizer = CharacterTokenizer(vocab_size=1024)
        examples = _build_synthetic_dataset(num_docs=50, seed=42, domain_weights={"math": 2.0, "coding": 1.0})
        texts = [ex.text for ex in examples]
        assert len(texts) == len(set(texts)), "Duplicate examples in synthetic dataset"
    
    def test_shared_unique_dataset(self):
        """Test SharedUniqueDataset creation."""
        from scripts.train_mini3_pilot import _build_synthetic_dataset, SharedUniqueDataset
        from src.data.tokenizer import CharacterTokenizer
        
        tokenizer = CharacterTokenizer(vocab_size=1024)
        shared = _build_synthetic_dataset(num_docs=10, seed=42)
        unique = _build_synthetic_dataset(num_docs=20, seed=43)
        
        dataset = SharedUniqueDataset(
            shared_examples=shared,
            unique_examples=unique,
            tokenizer=tokenizer,
            max_seq_len=64,
            shared_ratio=0.01,
        )
        assert len(dataset) > 0
        assert dataset.get_shared_ratio() > 0
    
    def test_model_config_defaults(self):
        """Test model config has expected defaults."""
        from scripts.train_mini3_pilot import build_model_config
        config = build_model_config()
        assert config["hidden_dim"] == 512
        assert config["num_layers"] == 12
        assert config["num_heads"] == 8
        assert config["vocab_size"] == 1024
    
    def test_model_config_custom(self):
        """Test model config accepts custom values."""
        from scripts.train_mini3_pilot import build_model_config
        config = build_model_config(vocab_size=2048, max_seq_len=512)
        assert config["vocab_size"] == 2048
        assert config["max_seq_len"] == 512
    
    def test_parameter_count(self):
        """Test parameter counting works."""
        import torch
        from scripts.train_mini3_pilot import build_model_config, build_model, count_parameters
        device = torch.device("cpu")
        config = build_model_config(vocab_size=1024, max_seq_len=128)
        model = build_model(config, device)
        params = count_parameters(model)
        assert params["total"] > 0
        assert params["trainable"] > 0
    
    def test_quality_lock_integration(self):
        """Test quality lock integration."""
        from scripts.train_mini3_pilot import QualityLock
        lock = QualityLock({"global_threshold": 0.965, "per_capability_threshold": 0.965})
        baseline = {"math": 0.9, "code": 0.9, "reasoning": 0.9}
        current = {"math": 0.92, "code": 0.91, "reasoning": 0.91}
        profile = lock.create_profile("test", baseline, current)
        assert profile.passes_all() is True
