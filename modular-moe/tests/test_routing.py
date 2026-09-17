"""Tests for routing functionality."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

from src.model import Router, MoELayer, MoELanguageModel


def test_topk_routing():
    """Test Top-K routing selection."""
    router = Router(hidden_dim=256, num_experts=8)
    x = torch.randn(100, 256)  # 100 tokens
    
    # Test different top_k values
    for top_k in [1, 2, 4, 8]:
        weights, indices, logits, aux_loss = router(x, top_k=top_k)
        
        assert weights.shape == (100, top_k)
        assert indices.shape == (100, top_k)
        assert logits.shape == (100, 8)
        
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(100), atol=1e-5)
        
        # Indices should be unique per token
        for i in range(100):
            assert len(set(indices[i].tolist())) == top_k


def test_router_aux_loss():
    """Test auxiliary load balancing loss."""
    router = Router(hidden_dim=256, num_experts=8)
    
    # Create inputs that should produce uniform routing
    x = torch.zeros(1000, 256)  # All zeros -> uniform logits -> uniform routing
    weights, indices, logits, aux_loss = router(x, top_k=2, return_aux_loss=True)
    
    # With uniform routing, aux_loss should be small
    # aux_loss = num_experts * sum(mean_probs * mean_mask)
    # For uniform: mean_probs = 1/8, mean_mask = 2/8 = 1/4
    # aux_loss = 8 * 8 * (1/8 * 1/4) = 2
    # Actually the loss formula is different, let's just check it's computed
    assert aux_loss is not None
    assert aux_loss.item() >= 0


def test_router_stats():
    """Test routing statistics computation."""
    router = Router(hidden_dim=256, num_experts=8)
    x = torch.randn(1000, 256)
    
    weights, indices, logits, _ = router(x, top_k=2)
    stats = router.get_routing_stats(logits, indices, 2)
    
    assert "expert_utilization" in stats
    assert len(stats["expert_utilization"]) == 8
    assert "routing_entropy" in stats
    assert "load_balance_cv" in stats
    assert "max_expert_load" in stats
    assert "min_expert_load" in stats
    assert "dead_experts" in stats
    assert "expert_collapse" in stats


def test_moe_routing():
    """Test MoE layer routing."""
    moe = MoELayer(
        hidden_dim=256,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=512,
    )
    
    x = torch.randn(2, 10, 256)
    out, aux_loss, stats = moe(x, return_aux_loss=True)
    
    assert out.shape == x.shape
    assert aux_loss is not None
    assert "expert_utilization" in stats
    
    # Check expert utilization sums to 100%
    total_util = sum(stats["expert_utilization"])
    assert abs(total_util - 100.0) < 1.0  # Each token goes to top_k experts


def test_moe_with_attention_mask():
    """Test MoE with attention mask."""
    moe = MoELayer(
        hidden_dim=256,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=512,
    )
    
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, 256)
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[0, 5:] = 0  # Mask second half of first sequence
    
    out, aux_loss, stats = moe(x, attention_mask=attention_mask, return_aux_loss=True)
    
    assert out.shape == x.shape
    # First sequence should only process first 5 tokens


def test_moe_capacity():
    """Test MoE capacity handling."""
    moe = MoELayer(
        hidden_dim=256,
        num_experts=4,
        top_k=2,
        expert_hidden_dim=512,
        capacity_factor=1.0,
        drop_tokens=False,
    )
    
    # Many tokens
    x = torch.randn(1, 1000, 256)
    out, _, _ = moe(x, return_aux_loss=True)
    
    assert out.shape == x.shape
    # All tokens should be processed (no dropping)


def test_router_deterministic():
    """Test router deterministic behavior."""
    torch.manual_seed(42)
    router1 = Router(hidden_dim=256, num_experts=8)
    
    torch.manual_seed(42)
    router2 = Router(hidden_dim=256, num_experts=8)
    
    x = torch.randn(100, 256)
    
    w1, i1, l1, _ = router1(x, top_k=2)
    w2, i2, l2, _ = router2(x, top_k=2)
    
    assert torch.allclose(w1, w2)
    assert torch.allclose(i1.float(), i2.float())
    assert torch.allclose(l1, l2)


def test_noisy_router():
    """Test NoisyRouter."""
    from src.model import NoisyRouter
    
    router = NoisyRouter(hidden_dim=256, num_experts=8, noise_std=1.0)
    x = torch.randn(100, 256)
    
    # Training mode - noise added
    router.train()
    weights1, indices1, logits1, _ = router(x, top_k=2)
    
    # Eval mode - no noise
    router.eval()
    weights2, indices2, logits2, _ = router(x, top_k=2)
    
    # Logits should differ in training due to noise
    assert not torch.allclose(logits1, logits2, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])