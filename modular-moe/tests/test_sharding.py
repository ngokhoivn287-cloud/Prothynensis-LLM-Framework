"""Tests for dataset sharding."""

import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from src.data.sharding import (
    shard_dataset,
    save_shard,
    compute_checksum,
    prepare_and_shard_dataset,
    verify_shards,
    get_shard_for_expert,
    ShardMetadata,
)
from src.data.tokenizer import CharacterTokenizer


def test_compute_checksum():
    """Test checksum computation."""
    data = np.array([1, 2, 3, 4, 5], dtype=np.uint16)
    checksum = compute_checksum(data)
    assert len(checksum) == 64  # SHA256 hex


def test_save_shard():
    """Test saving shard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "shard.bin"
        data = np.arange(100, dtype=np.uint16)
        
        checksum = save_shard(data, path)
        
        assert path.exists()
        loaded = np.fromfile(path, dtype=np.uint16)
        assert np.array_equal(loaded, data)
        assert compute_checksum(loaded) == checksum


def test_shard_dataset_sequential():
    """Test sequential sharding."""
    tokens = np.arange(1000, dtype=np.uint16)
    shards = shard_dataset(tokens, num_shards=10, shard_strategy="sequential")
    
    assert len(shards) == 10
    for shard in shards:
        assert len(shard) == 100
    
    # Check concatenation matches
    combined = np.concatenate(shards)
    assert np.array_equal(combined, tokens)


def test_shard_dataset_round_robin():
    """Test round-robin sharding."""
    tokens = np.arange(1000, dtype=np.uint16)
    shards = shard_dataset(
        tokens, num_shards=10, 
        expert_specific_ratio=1.0,  # All expert-specific
        num_experts=10,
        shard_strategy="round_robin",
    )
    
    assert len(shards) == 10
    # Each shard should have 100 tokens
    for shard in shards:
        assert len(shard) == 100
    
    # Check round-robin pattern
    # shard 0 should have tokens 0, 10, 20, ...
    assert shards[0][0] == 0
    assert shards[0][1] == 10


def test_shard_dataset_random():
    """Test random sharding."""
    tokens = np.arange(1000, dtype=np.uint16)
    shards = shard_dataset(
        tokens, num_shards=10,
        expert_specific_ratio=1.0,
        num_experts=10,
        shard_strategy="random",
        seed=42,
    )
    
    assert len(shards) == 10
    for shard in shards:
        assert len(shard) == 100
    
    # Combined should have all tokens (but shuffled)
    combined = np.concatenate(shards)
    assert len(combined) == 1000
    assert set(combined) == set(tokens)


def test_shard_dataset_expert_shared():
    """Test expert-specific + shared sharding."""
    tokens = np.arange(1000, dtype=np.uint16)
    shards = shard_dataset(
        tokens, num_shards=5,
        expert_specific_ratio=0.8,
        num_experts=4,
        shard_strategy="round_robin",
        seed=42,
    )
    
    # 4 expert shards + 1 shared shard
    assert len(shards) == 5
    
    # Expert shards: 800 tokens / 4 = 200 each
    for i in range(4):
        assert len(shards[i]) == 200
    
    # Shared shard: 200 tokens
    assert len(shards[4]) == 200


def test_prepare_and_shard_synthetic():
    """Test full prepare and shard pipeline with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "dataset_name": "synthetic",
                "synthetic_num_docs": 100,
                "max_doc_length": 100,
                "min_doc_length": 10,
                "tokenizer_type": "character",
                "tokenizer_vocab_size": 256,
                "num_shards": 4,
                "num_experts": 4,
                "expert_specific_ratio": 0.8,
                "shard_strategy": "round_robin",
                "shard_seed": 42,
            }
        }
        
        tokenizer = CharacterTokenizer(vocab_size=256)
        metadata = prepare_and_shard_dataset(config, tmpdir, tokenizer)
        
        assert len(metadata) == 4
        
        # Verify shards exist
        for meta in metadata:
            shard_path = Path(tmpdir) / meta.path
            assert shard_path.exists()
            
            # Verify checksum
            data = np.fromfile(shard_path, dtype=np.uint16)
            assert compute_checksum(data) == meta.checksum
            assert len(data) == meta.num_tokens
        
        # Verify metadata file
        meta_path = Path(tmpdir) / "metadata.json"
        assert meta_path.exists()


def test_verify_shards():
    """Test shard verification."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "dataset_name": "synthetic",
                "synthetic_num_docs": 100,
                "max_doc_length": 100,
                "min_doc_length": 10,
                "tokenizer_type": "character",
                "tokenizer_vocab_size": 256,
                "num_shards": 4,
                "num_experts": 4,
                "expert_specific_ratio": 0.8,
                "shard_strategy": "round_robin",
                "shard_seed": 42,
            }
        }
        
        tokenizer = CharacterTokenizer(vocab_size=256)
        prepare_and_shard_dataset(config, tmpdir, tokenizer)
        
        # Verify
        assert verify_shards(tmpdir) == True
        
        # Corrupt a shard
        shard_path = Path(tmpdir) / "shard_00000.bin"
        data = np.fromfile(shard_path, dtype=np.uint16)
        data[0] = 9999
        data.tofile(shard_path)
        
        # Should fail verification
        assert verify_shards(tmpdir) == False


def test_get_shard_for_expert():
    """Test getting shard for specific expert."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "data": {
                "dataset_name": "synthetic",
                "synthetic_num_docs": 100,
                "max_doc_length": 100,
                "min_doc_length": 10,
                "tokenizer_type": "character",
                "tokenizer_vocab_size": 256,
                "num_shards": 5,  # 4 expert + 1 shared
                "num_experts": 4,
                "expert_specific_ratio": 0.8,
                "shard_strategy": "round_robin",
                "shard_seed": 42,
            }
        }
        
        tokenizer = CharacterTokenizer(vocab_size=256)
        prepare_and_shard_dataset(config, tmpdir, tokenizer)
        
        # Get shard for expert 0 (includes shared)
        shards = get_shard_for_expert(tmpdir, expert_id=0, include_shared=True)
        
        assert len(shards) == 2  # expert_0 + shared
        
        # Get shard for expert 0 (expert only)
        shards_expert_only = get_shard_for_expert(tmpdir, expert_id=0, include_shared=False)
        assert len(shards_expert_only) == 1


def test_deterministic_sharding():
    """Test that sharding is deterministic."""
    tokens = np.arange(10000, dtype=np.uint16)
    
    # Round-robin is deterministic and seed-independent
    shards1 = shard_dataset(tokens, num_shards=8, shard_strategy="round_robin", seed=42)
    shards2 = shard_dataset(tokens, num_shards=8, shard_strategy="round_robin", seed=42)
    
    for s1, s2 in zip(shards1, shards2):
        assert np.array_equal(s1, s2)
    
    # Random strategy with same seed should be deterministic
    shards_r1 = shard_dataset(tokens, num_shards=8, shard_strategy="random", seed=42)
    shards_r2 = shard_dataset(tokens, num_shards=8, shard_strategy="random", seed=42)
    for s1, s2 in zip(shards_r1, shards_r2):
        assert np.array_equal(s1, s2)
    
    # Different seed with random strategy should produce different shards
    shards_r3 = shard_dataset(tokens, num_shards=8, shard_strategy="random", seed=123)
    assert not all(np.array_equal(s1, s3) for s1, s3 in zip(shards_r1, shards_r3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])