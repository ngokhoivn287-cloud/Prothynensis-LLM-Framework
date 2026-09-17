"""Tests for MoMMs ModelCoordinator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest

from momm.coordinator import ModelCoordinator, CoordinatorConfig
from momm.model import Model, ModelConfig, create_model
from momm.fusion import FusionConfig


def test_coordinator_basic_forward():
    """Test basic coordinator forward pass."""
    hidden_dim = 32
    model_configs = [
        ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
        ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
    ]
    coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=1)
    coordinator = ModelCoordinator([create_model(c) for c in model_configs], coord_config)

    x = torch.randn(2, 4, hidden_dim)
    out = coordinator(x, return_aux_loss=False)

    assert out["output"].shape == (2, 4, hidden_dim)
    assert out["routing_weights"].shape == (2 * 4, 1)
    assert out["model_indices"].shape == (2 * 4, 1)
    assert torch.allclose(out["routing_weights"].sum(dim=-1), torch.ones(8))


def test_coordinator_top_k_2():
    """Test coordinator with top_k=2."""
    hidden_dim = 32
    model_configs = [
        ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
        ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
    ]
    coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=2)
    coordinator = ModelCoordinator([create_model(c) for c in model_configs], coord_config)

    x = torch.randn(2, 4, hidden_dim)
    out = coordinator(x, return_aux_loss=False)

    assert out["output"].shape == (2, 4, hidden_dim)
    assert out["routing_weights"].shape == (8, 2)
    assert out["model_indices"].shape == (8, 2)
    assert torch.allclose(out["routing_weights"].sum(dim=-1), torch.ones(8))


def test_coordinator_attention_mask():
    """Test coordinator with attention mask."""
    hidden_dim = 32
    model_configs = [
        ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
        ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
    ]
    coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=1)
    coordinator = ModelCoordinator([create_model(c) for c in model_configs], coord_config)

    x = torch.randn(2, 4, hidden_dim)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
    out = coordinator(x, attention_mask=attention_mask, return_aux_loss=False)

    assert out["output"].shape == (2, 4, hidden_dim)


def test_coordinator_aux_loss():
    """Test coordinator auxiliary loss computation."""
    hidden_dim = 32
    model_configs = [
        ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
        ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
    ]
    coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=1, load_balancing=True, load_balancing_weight=0.1)
    coordinator = ModelCoordinator([create_model(c) for c in model_configs], coord_config)
    coordinator.train()

    x = torch.randn(2, 4, hidden_dim)
    out = coordinator(x, return_aux_loss=True)

    assert out["aux_loss"] is not None
    assert out["aux_loss"].shape == ()


def test_coordinator_routing_stats():
    """Test coordinator routing statistics."""
    hidden_dim = 32
    model_configs = [
        ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
        ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
    ]
    coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=1)
    coordinator = ModelCoordinator([create_model(c) for c in model_configs], coord_config)

    x = torch.randn(2, 4, hidden_dim)
    out = coordinator(x, return_aux_loss=False, return_routing_stats=True)

    assert "routing_stats" in out
    assert "model_utilization" in out["routing_stats"]
    assert len(out["routing_stats"]["model_utilization"]) == 2


def test_coordinator_from_config():
    """Test creating coordinator from configs."""
    hidden_dim = 32
    model_configs = [
        ModelConfig(model_id="m0", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
        ModelConfig(model_id="m1", hidden_dim=hidden_dim, num_layers=1, num_heads=4, head_dim=8, ffn_hidden_dim=64),
    ]
    coord_config = CoordinatorConfig(hidden_dim=hidden_dim, num_models=2, top_k=1)
    coordinator = ModelCoordinator.from_config(model_configs, coord_config)

    x = torch.randn(2, 4, hidden_dim)
    out = coordinator(x, return_aux_loss=False)
    assert out["output"].shape == (2, 4, hidden_dim)
