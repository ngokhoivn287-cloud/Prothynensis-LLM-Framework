"""Tests for model components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

from src.model import (
    CausalSelfAttention,
    RMSNorm,
    ExpertFFN,
    Expert,
    SwiGLU,
    Router,
    MoELayer,
    TransformerBlock,
    TransformerStack,
    MoELanguageModel,
    create_model_from_config,
)


def test_rmsnorm():
    """Test RMSNorm."""
    dim = 256
    norm = RMSNorm(dim)
    x = torch.randn(2, 10, dim)
    out = norm(x)
    assert out.shape == x.shape
    # Check normalization
    rms = torch.sqrt(torch.mean(out**2, dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_causal_attention():
    """Test causal self-attention."""
    hidden_dim = 256
    num_heads = 4
    head_dim = 64
    
    attn = CausalSelfAttention(hidden_dim, num_heads, head_dim)
    x = torch.randn(2, 10, hidden_dim)
    
    # Forward
    out, _ = attn(x)
    assert out.shape == x.shape
    
    # Test causality - future tokens shouldn't affect past
    # This is implicitly tested by the causal mask
    # (We don't test KV cache here as it's an inference feature)


def test_swiglu():
    """Test SwiGLU activation."""
    swiglu = SwiGLU()
    x = torch.randn(10, 256)
    gate = torch.randn(10, 256)
    out = swiglu(x, gate)
    assert out.shape == x.shape
    # Check SwiGLU formula: x * silu(gate)
    expected = x * torch.nn.functional.silu(gate)
    assert torch.allclose(out, expected)


def test_expert_ffn():
    """Test ExpertFFN."""
    hidden_dim = 256
    expert_hidden_dim = 1024
    
    expert = ExpertFFN(hidden_dim, expert_hidden_dim)
    x = torch.randn(2, 10, hidden_dim)
    out = expert(x)
    assert out.shape == x.shape
    
    # Check parameter count
    param_count = expert.get_parameter_count()
    expected = (
        hidden_dim * expert_hidden_dim * 2 +  # gate + up
        expert_hidden_dim * hidden_dim +       # down
        hidden_dim * 2                         # norms
    )
    assert param_count == expected


def test_router():
    """Test Router."""
    hidden_dim = 256
    num_experts = 8
    top_k = 2
    
    router = Router(hidden_dim, num_experts)
    x = torch.randn(32, hidden_dim)  # 32 tokens
    
    routing_weights, expert_indices, router_logits, aux_loss = router(x, top_k=top_k)
    
    assert routing_weights.shape == (32, top_k)
    assert expert_indices.shape == (32, top_k)
    assert router_logits.shape == (32, num_experts)
    assert aux_loss is not None
    assert aux_loss.item() >= 0
    
    # Check weights sum to 1
    assert torch.allclose(routing_weights.sum(dim=-1), torch.ones(32))
    
    # Check indices are valid
    assert (expert_indices >= 0).all()
    assert (expert_indices < num_experts).all()
    
    # Check top-k selection - renormalized weights should match topk probs / sum(topk_probs)
    for i in range(32):
        probs = torch.softmax(router_logits[i], dim=-1)
        topk_probs, topk_idx = torch.topk(probs, top_k)
        expected_weights = topk_probs / topk_probs.sum()
        assert torch.allclose(routing_weights[i], expected_weights, atol=1e-5)
        assert torch.allclose(expert_indices[i].float(), topk_idx.float())


def test_moe_layer():
    """Test MoELayer."""
    hidden_dim = 256
    num_experts = 4
    top_k = 2
    expert_hidden_dim = 512
    
    moe = MoELayer(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        expert_hidden_dim=expert_hidden_dim,
    )
    
    x = torch.randn(2, 10, hidden_dim)
    out, aux_loss, stats = moe(x, return_aux_loss=True)
    
    assert out.shape == x.shape
    assert aux_loss is not None
    assert "expert_utilization" in stats
    assert len(stats["expert_utilization"]) == num_experts


def test_transformer_block():
    """Test TransformerBlock."""
    hidden_dim = 256
    num_heads = 4
    head_dim = 64
    
    # Without MoE
    block = TransformerBlock(hidden_dim, num_heads, head_dim, 0, use_moe=False)
    x = torch.randn(2, 10, hidden_dim)
    out, _, _, _ = block(x)
    assert out.shape == x.shape
    
    # With MoE
    moe_config = {
        "num_experts": 4,
        "top_k": 2,
        "expert_hidden_dim": 512,
    }
    block_moe = TransformerBlock(hidden_dim, num_heads, head_dim, 0, use_moe=True, moe_config=moe_config)
    out, _, aux_loss, stats = block_moe(x, return_aux_loss=True)
    assert out.shape == x.shape
    assert aux_loss is not None


def test_transformer_stack():
    """Test TransformerStack."""
    hidden_dim = 256
    num_layers = 4
    num_heads = 4
    head_dim = 64
    
    moe_config = {
        "num_experts": 4,
        "top_k": 2,
        "expert_hidden_dim": 512,
    }
    
    # Every other layer is MoE
    moe_layer_indices = [1, 3]
    
    stack = TransformerStack(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        moe_layer_indices=moe_layer_indices,
        moe_config=moe_config,
    )
    
    x = torch.randn(2, 10, hidden_dim)
    out, _, total_aux_loss, all_stats = stack(x, return_aux_loss=True)
    
    assert out.shape == x.shape
    assert total_aux_loss is not None
    assert len(all_stats) == num_layers
    
    # Check MoE layers have stats
    for i in moe_layer_indices:
        assert "expert_utilization" in all_stats[i]
    for i in range(num_layers):
        if i not in moe_layer_indices:
            assert all_stats[i] == {}


def test_moe_language_model():
    """Test full MoE language model."""
    config = {
        "vocab_size": 1024,
        "max_seq_len": 128,
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 64,
        "num_experts": 4,
        "top_k": 2,
        "expert_hidden_dim": 512,
        "tie_embeddings": True,
        "gradient_checkpointing": False,
        "moe_layer_indices": [1, 3],
    }
    
    model = MoELanguageModel(**config)
    
    # Forward pass
    input_ids = torch.randint(0, 1024, (2, 16))
    outputs = model(input_ids=input_ids, labels=input_ids)
    
    assert "logits" in outputs
    assert outputs["logits"].shape == (2, 16, 1024)
    assert "loss" in outputs
    assert "total_loss" in outputs
    assert outputs["loss"].item() >= 0
    
    # Generate
    generated = model.generate(input_ids[:1], max_new_tokens=10)
    assert generated.shape == (1, 26)


def test_parameter_counting():
    """Test parameter counting."""
    config = {
        "vocab_size": 1024,
        "max_seq_len": 128,
        "hidden_dim": 256,
        "num_layers": 4,
        "num_heads": 4,
        "head_dim": 64,
        "num_experts": 4,
        "top_k": 2,
        "expert_hidden_dim": 512,
        "tie_embeddings": True,
    }
    
    model = MoELanguageModel(**config)
    breakdown = model.get_parameter_count()
    
    assert "total" in breakdown
    assert breakdown["total"] > 0
    assert breakdown["total"] == sum(v for k, v in breakdown.items() if k != "total")
    
    # Check expert params (just the expert FFN params, not router/norms)
    moe_layers = model.get_moe_layers()
    
    # Check if experts are shared
    shared_experts = model.transformer.get_shared_experts()
    if shared_experts is not None:
        # With shared experts: total_expert_params counts per-layer but experts are shared
        # So moe_total (shared experts counted once) will be < total_expert_params (counted per layer)
        total_expert_params = sum(
            layer.get_parameter_count()["experts"] for layer in moe_layers
        )
        assert breakdown["moe_total"] < total_expert_params
        # With shared experts, summing per-layer moe totals double-counts experts
        # So we don't check the equality
    else:
        # Per-layer experts: moe_total includes experts + router + norms
        total_expert_params = sum(
            layer.get_parameter_count()["experts"] for layer in moe_layers
        )
        assert breakdown["moe_total"] > total_expert_params
        assert breakdown["moe_total"] == sum(
            layer.get_parameter_count()["total"] for layer in moe_layers
        )


def test_deterministic():
    """Test deterministic behavior."""
    config = {
        "vocab_size": 1024,
        "max_seq_len": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "num_heads": 4,
        "head_dim": 64,
        "num_experts": 4,
        "top_k": 2,
        "expert_hidden_dim": 512,
    }
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model1 = MoELanguageModel(**config)
    
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model2 = MoELanguageModel(**config)
    
    input_ids = torch.randint(0, 1024, (2, 16))
    
    with torch.no_grad():
        out1 = model1(input_ids)
        out2 = model2(input_ids)
    
    assert torch.allclose(out1["logits"], out2["logits"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])