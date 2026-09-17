"""Real inference test for Prothynesis Runtime."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from momm.model import Model, ModelConfig, create_model
from momm.coordinator import ModelCoordinator, CoordinatorConfig
from momm.fusion import FusionLayer, FusionConfig, create_fusion
from src.runtime.controller import ProthynesisRuntime
from src.runtime.settings import SettingsManager


class TestRealInference:
    def test_single_model_inference(self):
        """Test real inference with a single model."""
        config = ModelConfig(
            model_id="test_inference",
            hidden_dim=32,
            num_layers=1,
            num_heads=4,
            head_dim=8,
            ffn_hidden_dim=64,
            vocab_size=100,
            max_seq_len=64,
        )
        model = create_model(config)
        model.eval()
        
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        with torch.no_grad():
            out = model(input_ids)
        
        assert out["logits"].shape == (1, 5, 100)
    
    def test_coordinator_inference(self):
        """Test real inference through the coordinator."""
        hidden_dim = 32
        model_configs = [
            ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64, vocab_size=100),
            ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64, vocab_size=100),
        ]
        coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=1)
        coordinator = ModelCoordinator([create_model(c) for c in model_configs], coord_config)
        coordinator.eval()
        
        hidden_states = torch.randn(1, 5, hidden_dim)
        with torch.no_grad():
            out = coordinator(hidden_states, return_aux_loss=False)
        
        assert out["output"].shape == (1, 5, hidden_dim)
    
    def test_runtime_controller_creation(self):
        """Test runtime controller can be created."""
        runtime = ProthynesisRuntime()
        assert runtime.registry is not None
        assert runtime.inference_engine is not None
        runtime.shutdown()
    
    def test_settings_persistence(self, tmp_path):
        """Test settings are persisted correctly."""
        settings = SettingsManager(tmp_path / "settings.yaml")
        settings.set("runtime.backend", "cuda")
        settings.set("generation.temperature", 0.8)
        
        # Reload
        settings2 = SettingsManager(tmp_path / "settings.yaml")
        assert settings2.get("runtime.backend") == "cuda"
        assert settings2.get("generation.temperature") == 0.8
    
    def test_hardware_detection_returns_valid_data(self):
        """Test hardware detection returns valid data."""
        from src.runtime.hardware import detect_hardware
        report = detect_hardware()
        assert report.platform != ""
        assert report.cpu.cores >= 1
        assert report.recommended_backend in {"cpu", "cuda"}
