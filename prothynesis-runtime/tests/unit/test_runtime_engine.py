import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "modular-moe" / "src"))

from prothynesis_runtime.runtime.engine import (
    ModelLoadError,
    UnsupportedFormatError,
    _build_model_from_checkpoint,
    _infer_config_from_state,
    load_checkpoint,
    load_model_from_pmo,
)
from prothynesis_runtime.runtime.orchestrator import InferenceEngine, RuntimeOrchestrator


def _create_test_checkpoint(path: Path, vocab_size: int = 1024, hidden_dim: int = 256, num_layers: int = 2) -> None:
    from model import MoELanguageModel

    config = {
        "vocab_size": vocab_size,
        "max_seq_len": 128,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": 4,
        "head_dim": 64,
        "num_experts": 1,
        "top_k": 1,
        "expert_hidden_dim": hidden_dim * 2,
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
    torch.save(model.state_dict(), path)


def test_load_checkpoint_returns_state_dict():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _create_test_checkpoint(tmp_path)
        state = load_checkpoint(tmp_path)
        assert "token_embedding.weight" in state
        assert "transformer.layers.0.attn_norm.weight" in state
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_load_checkpoint_missing_file():
    with pytest.raises(ModelLoadError, match="Checkpoint not found"):
        load_checkpoint(Path("/nonexistent/model.pt"))


def test_infer_config_from_state():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _create_test_checkpoint(tmp_path, vocab_size=2048, hidden_dim=512, num_layers=4)
        state = torch.load(tmp_path, map_location="cpu", weights_only=False)
        config = _infer_config_from_state(state)
        assert config["vocab_size"] == 2048
        assert config["hidden_dim"] == 512
        assert config["num_layers"] == 4
        assert config["max_seq_len"] == 128
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_build_model_from_checkpoint():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _create_test_checkpoint(tmp_path)
        model, _ = _build_model_from_checkpoint(tmp_path, device="cpu")
        assert model is not None
        x = torch.randint(0, 1024, (1, 8))
        with torch.no_grad():
            out = model(x)
        assert "logits" in out
        assert out["logits"].shape == (1, 8, 1024)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_inference_engine_load_and_generate():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _create_test_checkpoint(tmp_path)
        engine = InferenceEngine(model_path=str(tmp_path))
        engine.load(str(tmp_path))
        assert engine.loaded is True
        response = engine.generate("Hello world")
        assert isinstance(response, str)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_inference_engine_missing_model():
    engine = InferenceEngine(model_path="/nonexistent/model.pt")
    with pytest.raises(ModelLoadError, match="Model path not found"):
        engine.load("/nonexistent/model.pt")


def test_inference_engine_unload():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _create_test_checkpoint(tmp_path)
        engine = InferenceEngine(model_path=str(tmp_path))
        engine.load(str(tmp_path))
        assert engine.loaded is True
        engine.unload()
        assert engine.loaded is False
        assert engine._model is None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_runtime_orchestrator_with_real_model():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _create_test_checkpoint(tmp_path)
        orch = RuntimeOrchestrator()
        assert orch.inference_engine.loaded is False
        orch.inference_engine.load(str(tmp_path))
        assert orch.inference_engine.loaded is True
        result = orch.chat([{"role": "user", "content": "Hello"}], model_id="test")
        assert "choices" in result
        assert result["object"] == "chat.completion"
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_runtime_orchestrator_chat_without_model():
    orch = RuntimeOrchestrator()
    with pytest.raises(ModelLoadError):
        orch.chat([{"role": "user", "content": "Hello"}], model_id="test")
