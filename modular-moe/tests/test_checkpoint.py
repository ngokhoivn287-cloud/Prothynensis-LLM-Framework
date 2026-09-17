"""Tests for checkpointing."""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

from src.training.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    atomic_save,
    atomic_save_safetensors,
)
from src.model import ExpertFFN, MoELanguageModel


def test_atomic_save():
    """Test atomic save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.pt"
        
        data = {"key": torch.tensor([1, 2, 3])}
        atomic_save(data, path)
        
        assert path.exists()
        loaded = torch.load(path)
        assert torch.equal(loaded["key"], data["key"])


def test_atomic_save_safetensors():
    """Test atomic save with safetensors."""
    import platform
    if platform.system() == "Windows":
        pytest.skip("safetensors file locking issue on Windows")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.safetensors"
        
        tensors = {"weight": torch.randn(10, 10), "bias": torch.randn(10)}
        atomic_save_safetensors(tensors, path)
        
        assert path.exists()
        import safetensors.torch
        loaded = safetensors.torch.load_file(str(path))
        assert torch.allclose(loaded["weight"], tensors["weight"])
        assert torch.allclose(loaded["bias"], tensors["bias"])


def test_checkpoint_manager():
    """Test CheckpointManager."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            root_dir=tmpdir,
            format="safetensors",
            keep_last_n=2,
        )
        
        model = ExpertFFN(256, 512)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Save first checkpoint
        meta1 = manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=100,
            tokens_seen=10000,
            config={"test": "config"},
            metrics={"loss": 2.5},
        )
        
        assert meta1.exists()
        assert (meta1 / "model.safetensors").exists()
        assert (meta1 / "optimizer.pt").exists()
        assert (meta1 / "scheduler.pt").exists()
        assert (meta1 / "rng.pt").exists()
        assert (meta1 / "metadata.json").exists()
        
        # Save second checkpoint
        meta2 = manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=200,
            tokens_seen=20000,
            metrics={"loss": 2.3},
        )
        
        # Save third checkpoint (should rotate first)
        meta3 = manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=300,
            tokens_seen=30000,
            metrics={"loss": 2.1},
        )
        
        # First should be rotated out
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 2
        assert checkpoints[0].name == "step_00000200"
        assert checkpoints[1].name == "step_00000300"


def test_checkpoint_load():
    """Test checkpoint loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(root_dir=tmpdir, format="safetensors")
        
        model = ExpertFFN(256, 512)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        
        # Save
        checkpoint_dir = manager.save(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=500,
            tokens_seen=50000,
            config={"lr": 1e-3},
            metrics={"loss": 2.0, "ppl": 7.4},
        )
        
        # Create new model and optimizer
        model2 = ExpertFFN(256, 512)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10)
        
        # Load
        metadata = manager.load(
            checkpoint_path=checkpoint_dir,
            model=model2,
            optimizer=optimizer2,
            scheduler=scheduler2,
        )
        
        assert metadata.step == 500
        assert metadata.tokens_seen == 50000
        assert metadata.metrics["loss"] == 2.0
        
        # Check model weights match
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2)
        
        # Check optimizer state matches
        for g1, g2 in zip(optimizer.param_groups, optimizer2.param_groups):
            assert g1["lr"] == g2["lr"]


def test_checkpoint_rng():
    """Test RNG state checkpointing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(root_dir=tmpdir, save_rng=True)
        
        model = ExpertFFN(256, 512)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Set specific RNG state
        torch.manual_seed(12345)
        torch.cuda.manual_seed_all(12345)
        
        # Generate some random numbers
        _ = torch.randn(10)
        _ = torch.randn(10)
        
        # Save
        checkpoint_dir = manager.save(
            model=model,
            optimizer=optimizer,
            step=100,
        )
        
        # Generate more
        r1 = torch.randn(10)
        
        # Load
        model2 = ExpertFFN(256, 512)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
        manager.load(
            checkpoint_path=checkpoint_dir,
            model=model2,
            optimizer=optimizer2,
            load_rng=True,
        )
        
        # Generate - should match r1
        r2 = torch.randn(10)
        assert torch.allclose(r1, r2)


def test_full_model_checkpoint():
    """Test checkpointing full MoE model."""
    with tempfile.TemporaryDirectory() as tmpdir:
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
        }
        
        model = MoELanguageModel(**config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        manager = CheckpointManager(root_dir=tmpdir, format="safetensors")
        
        checkpoint_dir = manager.save(
            model=model,
            optimizer=optimizer,
            step=1000,
            tokens_seen=1000000,
            config=config,
            metrics={"loss": 3.2, "ppl": 24.5},
        )
        
        # Load into new model
        model2 = MoELanguageModel(**config)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        metadata = manager.load(
            checkpoint_path=checkpoint_dir,
            model=model2,
            optimizer=optimizer2,
        )
        
        assert metadata.step == 1000
        
        # Verify outputs match
        input_ids = torch.randint(0, 1024, (2, 16))
        with torch.no_grad():
            out1 = model(input_ids)
            out2 = model2(input_ids)
        
        assert torch.allclose(out1["logits"], out2["logits"])


def test_expert_checkpoint():
    """Test expert-only checkpoint."""
    from src.training.checkpoint import save_expert_checkpoint, load_expert_checkpoint
    
    with tempfile.TemporaryDirectory() as tmpdir:
        expert = ExpertFFN(256, 512)
        optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
        
        path = save_expert_checkpoint(
            expert=expert,
            optimizer=optimizer,
            scheduler=None,
            step=100,
            tokens_seen=10000,
            dataset_position=0,
            config={"test": "config"},
            metrics={"loss": 2.5},
            output_dir=tmpdir,
            expert_id=0,
        )
        
        assert os.path.exists(path)
        
        # Load
        expert2 = ExpertFFN(256, 512)
        optimizer2 = torch.optim.AdamW(expert2.parameters(), lr=1e-3)
        
        metadata = load_expert_checkpoint(
            expert=expert2,
            optimizer=optimizer2,
            scheduler=None,
            checkpoint_dir=os.path.dirname(path),
        )
        
        assert metadata.step == 100
        
        # Check weights match
        for p1, p2 in zip(expert.parameters(), expert2.parameters()):
            assert torch.allclose(p1, p2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])