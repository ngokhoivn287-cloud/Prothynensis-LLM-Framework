from __future__ import annotations

import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from prothynesis_runtime.pmo.package import PMOPackage


class SimpleCharacterTokenizer:
    """Minimal character-level tokenizer for runtime inference."""

    def __init__(self, vocab_size: int = 1024):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3

    def encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        for ch in text[: self.vocab_size - 10]:
            ids.append(ord(ch) % (self.vocab_size - 10) + 10)
        ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: list[int] | int) -> str:
        if isinstance(ids, int):
            ids = [ids]
        chars = []
        for tid in ids:
            if tid in (self.pad_token_id, self.bos_token_id, self.eos_token_id):
                continue
            chars.append(chr(tid - 10))
        return "".join(chars)


_MODULAR_MOE_ROOT = Path(__file__).resolve().parents[4] / "modular-moe"
if str(_MODULAR_MOE_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODULAR_MOE_ROOT))


class ModelLoadError(Exception):
    """Raised when a model cannot be loaded."""


class UnsupportedFormatError(ModelLoadError):
    """Raised when the model format is not supported."""


def _resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_safetensors(path: Path, device: str) -> Dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load_file
        return load_file(str(path), device=device)
    except ImportError as exc:
        raise ModelLoadError("safetensors is required to load .safetensors files") from exc


def _load_torch(path: Path, device: str) -> Dict[str, torch.Tensor]:
    state = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return state


def load_checkpoint(path: Union[str, Path], device: str = "auto") -> Dict[str, torch.Tensor]:
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"Checkpoint not found: {path}")
    device = _resolve_device(device)
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return _load_safetensors(path, device)
    if suffix in (".pt", ".bin", ".pth"):
        return _load_torch(path, device)
    raise UnsupportedFormatError(f"Unsupported checkpoint format: {suffix}")


def load_model_from_pmo(
    pmo_path: Union[str, Path],
    device: str = "auto",
    config: Optional[Dict[str, Any]] = None,
) -> tuple[Any, Dict[str, Any]]:
    """Load a model from a PMO package.

    Returns (model, metadata).
    """
    pmo = PMOPackage(str(pmo_path))
    if not pmo.verify():
        raise ModelLoadError(f"Invalid PMO package: {pmo_path}")

    with pmo.path.open("rb") as f:
        import zipfile
        with zipfile.ZipFile(f, "r") as zf:
            names = zf.namelist()

    meta = {}
    manifest = {}
    if "metadata.json" in names:
        with zipfile.ZipFile(str(pmo.path), "r") as zf:
            meta = json.loads(zf.read("metadata.json").decode("utf-8"))
    if "manifest.json" in names:
        with zipfile.ZipFile(str(pmo.path), "r") as zf:
            manifest = json.loads(zf.read("manifest.json").decode("utf-8"))

    model_index = {}
    if "model_index.json" in names:
        with zipfile.ZipFile(str(pmo.path), "r") as zf:
            model_index = json.loads(zf.read("model_index.json").decode("utf-8"))

    model = None
    checkpoint_path = None
    for entry in model_index.get("models", []):
        cp = entry.get("checkpoint_path") or entry.get("path")
        if cp:
            checkpoint_path = cp
            break

    if checkpoint_path is None:
        for name in names:
            if name.endswith((".safetensors", ".bin", ".pt", ".pth")):
                checkpoint_path = name
                break

    if checkpoint_path is None:
        raise ModelLoadError("PMO package does not contain a model checkpoint")

    with zipfile.ZipFile(str(pmo.path), "r") as zf:
        try:
            data = zf.read(checkpoint_path)
        except KeyError as exc:
            raise ModelLoadError(f"Checkpoint not found in PMO: {checkpoint_path}") from exc

    suffix = Path(checkpoint_path).suffix.lower()
    tmp_path = Path(f"_pmo_extracted_{Path(pmo.path).stem}{suffix}")
    tmp_path.write_bytes(data)

    try:
        model, vocab_size = _build_model_from_checkpoint(tmp_path, device, config=config)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return model, vocab_size, {"pmo_metadata": meta, "manifest": manifest}


def _build_model_from_checkpoint(
    checkpoint_path: Path,
    device: str,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[Any, int]:
    try:
        from model import MoELanguageModel
    except ImportError as exc:
        raise ModelLoadError("modular-moe is required to load models") from exc

    device = _resolve_device(device)
    state = load_checkpoint(checkpoint_path, device=device)

    if config is None:
        config = _infer_config_from_state(state)

    vocab_size = config.get("vocab_size", 1024)
    model = MoELanguageModel(**config)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, vocab_size


def _infer_config_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    embed_key = "token_embedding.weight"
    if embed_key not in state:
        raise ModelLoadError("Cannot infer model config from checkpoint: missing token_embedding.weight")

    vocab_size, hidden_dim = state[embed_key].shape

    num_layers = 0
    while f"transformer.layers.{num_layers}.attn_norm.weight" in state:
        num_layers += 1

    if num_layers == 0:
        raise ModelLoadError("Cannot infer model config from checkpoint: no transformer blocks found")

    head_dim = hidden_dim // 8
    if head_dim == 0:
        head_dim = 64
    num_heads = hidden_dim // head_dim

    max_seq_len = 2048
    pos_key = "position_embedding.weight"
    if pos_key in state:
        max_seq_len = state[pos_key].shape[0]

    num_experts = 1
    for key in state.keys():
        if "experts." in key:
            parts = key.split(".")
            for part in parts:
                if part.isdigit():
                    num_experts = max(num_experts, int(part) + 1)
            break

    return {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "num_experts": num_experts,
        "top_k": min(1, num_experts),
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
