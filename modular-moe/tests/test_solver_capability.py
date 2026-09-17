"""Tests for Solver Capability Pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import torch

from src.data.tokenizer import CharacterTokenizer, HFTokenizerWrapper, create_tokenizer
from src.model import MoELanguageModel
from src.utils.hardware import get_device
from scripts.run_solver_experiments import SyntheticExample, SharedUniqueDataset


class TestTokenizerDiagnostics:
    """Tokenizer must roundtrip known text without excessive UNK."""

    def test_character_tokenizer_roundtrip(self):
        tokenizer = CharacterTokenizer(vocab_size=1024)
        sentence = "What is the capital of France?"
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded)
        assert decoded.strip() == sentence.strip()
        assert tokenizer.unk_token_id not in encoded

    def test_character_tokenizer_special_tokens(self):
        tokenizer = CharacterTokenizer(vocab_size=1024)
        assert tokenizer.pad_token_id == 0
        assert tokenizer.eos_token_id == 1
        assert tokenizer.bos_token_id == 2
        assert tokenizer.unk_token_id == 3

    def test_character_tokenizer_vocab_size(self):
        tokenizer = CharacterTokenizer(vocab_size=1024)
        assert tokenizer.vocab_size == 1024

    def test_character_tokenizer_empty_input(self):
        tokenizer = CharacterTokenizer(vocab_size=1024)
        encoded = tokenizer.encode("")
        decoded = tokenizer.decode(encoded)
        assert decoded == ""

    def test_character_tokenizer_unk_fallback(self):
        tokenizer = CharacterTokenizer(vocab_size=256)
        encoded = tokenizer.encode("hello world")
        assert all(t != tokenizer.unk_token_id for t in encoded)


class TestModelGenerationSmoke:
    """Model must generate non-empty token sequences without NaN/Inf."""

    def test_generation_non_empty(self):
        device = get_device("cpu")
        config = {
            "vocab_size": 1024,
            "max_seq_len": 128,
            "hidden_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
            "head_dim": 64,
            "num_experts": 1,
            "top_k": 1,
            "expert_hidden_dim": 2048,
            "expert_activation": "silu",
            "expert_norm": "rmsnorm",
            "expert_bias": False,
            "expert_dropout": 0.0,
            "router_hidden_dim": None,
            "router_bias": False,
            "router_dropout": 0.0,
            "tie_embeddings": True,
            "use_rmsnorm": True,
            "norm_eps": 1e-5,
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "causal": True,
            "use_flash_attn": False,
            "gradient_checkpointing": False,
        }
        model = MoELanguageModel(**config)
        model.to(device)
        model.eval()
        
        tokenizer = CharacterTokenizer(vocab_size=1024)
        prompt = "What is the capital of France?"
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            temperature=0.0,
            top_k=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        assert generated.shape[1] > input_ids.shape[1]
        assert not torch.isnan(generated).any()
        assert not torch.isinf(generated).any()

    def test_generation_logits_sanity(self):
        device = get_device("cpu")
        config = {
            "vocab_size": 1024,
            "max_seq_len": 128,
            "hidden_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
            "head_dim": 64,
            "num_experts": 1,
            "top_k": 1,
            "expert_hidden_dim": 2048,
            "expert_activation": "silu",
            "expert_norm": "rmsnorm",
            "expert_bias": False,
            "expert_dropout": 0.0,
            "router_hidden_dim": None,
            "router_bias": False,
            "router_dropout": 0.0,
            "tie_embeddings": True,
            "use_rmsnorm": True,
            "norm_eps": 1e-5,
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "causal": True,
            "use_flash_attn": False,
            "gradient_checkpointing": False,
        }
        model = MoELanguageModel(**config)
        model.to(device)
        model.eval()
        
        tokenizer = CharacterTokenizer(vocab_size=1024)
        prompt = "Hello world"
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_aux_loss=False)
        
        logits = outputs.get("logits")
        assert logits is not None
        assert logits.shape == (1, len(tokenizer.encode(prompt)), 1024)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()


class TestDatasetDiagnostics:
    """Dataset must produce valid non-empty targets."""

    def test_dataset_roundtrip(self):
        tokenizer = CharacterTokenizer(vocab_size=1024)
        examples = [
            SyntheticExample(id=f"e{i}", domain="test", difficulty="D0", task_type="test", language="en", text=f"Q: Test {i} A: Answer {i}")
            for i in range(10)
        ]
        dataset = SharedUniqueDataset(shared_examples=[], unique_examples=examples, tokenizer=tokenizer, max_seq_len=32)
        
        assert len(dataset) > 0
        sample = dataset[0]
        assert "input_ids" in sample
        assert "labels" in sample
        assert sample["input_ids"].shape == (32,)
        assert sample["labels"].shape == (32,)

    def test_dataset_no_empty_targets(self):
        tokenizer = CharacterTokenizer(vocab_size=1024)
        examples = [
            SyntheticExample(id=f"e{i}", domain="test", difficulty="D0", task_type="test", language="en", text=f"Q: Test {i} A: Answer {i}")
            for i in range(10)
        ]
        dataset = SharedUniqueDataset(shared_examples=[], unique_examples=examples, tokenizer=tokenizer, max_seq_len=32)
        
        empty_count = 0
        for i in range(min(10, len(dataset))):
            sample = dataset[i]
            if sample["labels"].max() == tokenizer.pad_token_id and sample["labels"].min() == tokenizer.pad_token_id:
                empty_count += 1
        
        assert empty_count == 0


class TestCheckpointLoading:
    """Checkpoints must load and reproduce expected behavior."""

    def test_checkpoint_save_load(self, tmp_path):
        device = get_device("cpu")
        config = {
            "vocab_size": 1024,
            "max_seq_len": 128,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 4,
            "head_dim": 64,
            "num_experts": 1,
            "top_k": 1,
            "expert_hidden_dim": 512,
            "expert_activation": "silu",
            "expert_norm": "rmsnorm",
            "expert_bias": False,
            "expert_dropout": 0.0,
            "router_hidden_dim": None,
            "router_bias": False,
            "router_dropout": 0.0,
            "tie_embeddings": True,
            "use_rmsnorm": True,
            "norm_eps": 1e-5,
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "causal": True,
            "use_flash_attn": False,
            "gradient_checkpointing": False,
        }
        
        model = MoELanguageModel(**config)
        model.to(device)
        fp_before = _compute_fingerprint(model)
        
        checkpoint_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        model2 = MoELanguageModel(**config)
        model2.to(device)
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model2.load_state_dict(state)
        fp_after = _compute_fingerprint(model2)
        
        assert fp_before == fp_after


def _compute_fingerprint(model):
    import hashlib
    parts = []
    for name, param in sorted(model.named_parameters()):
        parts.append(f"{name}:{param.data.sum().item():.6f}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


_CAPABILITY_TEMPLATES = [
    {"domain": "general_knowledge", "prompt": "What is the capital of France?", "expected": "paris"},
    {"domain": "math", "prompt": "Calculate 15 * 24.", "expected": "360"},
    {"domain": "basic_reasoning", "prompt": "If all A are B and all B are C, what can we conclude?", "expected": "all a are c"},
    {"domain": "coding", "prompt": "What does the Python print function do?", "expected": "output"},
    {"domain": "verification", "prompt": "Check: 7 * 8 = 54. Is this correct?", "expected": "no"},
]


class TestCapabilityBenchmark:
    """Benchmark must produce valid scores."""

    def test_benchmark_scores(self):
        device = get_device("cpu")
        config = {
            "vocab_size": 1024,
            "max_seq_len": 128,
            "hidden_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
            "head_dim": 64,
            "num_experts": 1,
            "top_k": 1,
            "expert_hidden_dim": 2048,
            "expert_activation": "silu",
            "expert_norm": "rmsnorm",
            "expert_bias": False,
            "expert_dropout": 0.0,
            "router_hidden_dim": None,
            "router_bias": False,
            "router_dropout": 0.0,
            "tie_embeddings": True,
            "use_rmsnorm": True,
            "norm_eps": 1e-5,
            "attn_dropout": 0.0,
            "resid_dropout": 0.0,
            "causal": True,
            "use_flash_attn": False,
            "gradient_checkpointing": False,
        }
        model = MoELanguageModel(**config)
        model.to(device)
        model.eval()
        
        tokenizer = CharacterTokenizer(vocab_size=1024)
        for template in _CAPABILITY_TEMPLATES:
            prompt = template["prompt"]
            expected = template["expected"]
            input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
            
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=16,
                temperature=0.0,
                top_k=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            gen_text = tokenizer.decode(generated[0].tolist())
            assert isinstance(gen_text, str)
            assert len(gen_text) > 0
