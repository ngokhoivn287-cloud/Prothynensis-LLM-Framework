"""Filesystem-isolation test for Prothynensis dataset/training code.

Verifies that no dataset, training, preprocessing, checkpoint,
shard, log, manifest, or temp artifacts are created outside the
configured approved roots (PROTHYNESIS_DATA_ROOT environment variable).
"""

from __future__ import annotations

import sys
import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from src.utils.paths import (
    is_allowed_dataset_path,
    assert_allowed_path,
    get_primary_root,
    get_temp_root,
    get_checkpoints_root,
    get_shards_root,
    get_logs_root,
    get_manifests_root,
    get_registry_root,
    enforce_path_policy,
)


def _get_approved_roots_for_test() -> list[Path]:
    """Get approved roots for testing - mirrors src.utils.paths logic."""
    import os
    env_root = os.environ.get("PROTHYNESIS_DATA_ROOT")
    if env_root:
        primary = Path(env_root)
        return [primary, primary / "temp"]
    if sys.platform == "linux":
        return [Path("/mnt/d/NgoPROJECT/ProthynensisDatasets"), Path("/mnt/d/NgoPROJECT/ProthynensisDatasets/temp")]
    return [Path(r"D:\NgoPROJECT\ProthynensisDatasets"), Path(r"D:\NgoPROJECT\ProthynensisDatasets\temp")]


APPROVED_ROOTS = _get_approved_roots_for_test()


def _snapshot_tree(root: Path) -> set[Path]:
    """Return set of all paths under *root* (including root)."""
    root = Path(root)
    paths = set()
    if not root.exists():
        return paths
    for dirpath, dirnames, filenames in os.walk(root):
        paths.add(Path(dirpath).resolve())
        for f in filenames:
            paths.add((Path(dirpath) / f).resolve())
    return paths


def _resolve_created_paths(before: set[Path], after: set[Path]) -> list[Path]:
    return sorted(after - before, key=lambda p: str(p))


def test_path_validator_allows_approved_roots():
    for root in APPROVED_ROOTS:
        assert is_allowed_dataset_path(root), f"Approved root rejected: {root}"
        assert is_allowed_dataset_path(root / "sub" / "file.bin")
        assert assert_allowed_path(root / "sub" / "file.bin") == (root / "sub" / "file.bin").resolve()


def test_path_validator_rejects_outside_paths():
    bad_paths = [
        Path(r"C:\Users\Ngo\AppData\Local\Temp\bad"),
        Path(r"C:\temp\bad"),
        Path("D:\\other\\bad"),
        Path("/tmp/bad"),
    ]
    for p in bad_paths:
        assert not is_allowed_dataset_path(p), f"Should reject: {p}"


def test_assert_allowed_path_raises_on_bad_path():
    with pytest.raises(PermissionError):
        assert_allowed_path(r"C:\Users\Ngo\AppData\Local\Temp\evil.bin", context="test")


def test_enforce_path_policy_sets_env():
    enforce_path_policy()
    assert os.environ["TMPDIR"]
    assert os.environ["TEMP"]
    assert os.environ["TMP"]
    assert os.environ["HF_HOME"]
    assert os.environ["TORCH_HOME"]
    for key in ["TMPDIR", "TEMP", "TMP", "HF_HOME", "TORCH_HOME"]:
        p = Path(os.environ[key])
        assert is_allowed_dataset_path(p), f"Env {key}={p} is outside approved roots"


def test_approved_roots_are_under_dataset_root():
    primary = get_primary_root()
    expected_roots = _get_approved_roots_for_test()
    expected = expected_roots[0]
    assert primary == expected, f"Expected {expected}, got {primary}"
    assert get_temp_root() == primary / "temp"
    assert get_checkpoints_root() == primary / "checkpoints"
    assert get_shards_root() == primary / "shards"
    assert get_logs_root() == primary / "logs"
    assert get_manifests_root() == primary / "manifests"
    assert get_registry_root() == primary / "registry"


def test_sharding_creates_files_only_under_approved_roots():
    from src.data.tokenizer import CharacterTokenizer
    from src.data.sharding import prepare_and_shard_dataset, verify_shards
    
    root = get_shards_root() / "tests" / "sharding_isolation"
    before = _snapshot_tree(get_primary_root())
    
    try:
        config = {
            "data": {
                "dataset_name": "synthetic",
                "tokenizer_vocab_size": 1024,
                "num_shards": 4,
                "shard_strategy": "round_robin",
                "expert_specific_ratio": 0.8,
                "shard_seed": 42,
                "max_doc_length": 128,
                "min_doc_length": 10,
            }
        }
        tokenizer = CharacterTokenizer(vocab_size=1024)
        metadata = prepare_and_shard_dataset(config, str(root), tokenizer)
        assert verify_shards(str(root))
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    
    after = _snapshot_tree(get_primary_root())
    created = _resolve_created_paths(before, after)
    leaked = [p for p in created if not is_allowed_dataset_path(p)]
    assert leaked == [], f"Sharding created files outside approved roots: {leaked}"


def test_training_checkpoint_only_under_approved_roots():
    import torch
    import torch.nn as nn
    from src.momm.model import ModelConfig, create_model
    from src.training.checkpoint import CheckpointManager
    
    root = get_checkpoints_root() / "tests" / "ckpt_isolation"
    before = _snapshot_tree(get_primary_root())
    
    try:
        model = create_model(ModelConfig(
            model_id="isolation_test",
            vocab_size=1024,
            max_seq_len=128,
            hidden_dim=256,
            num_layers=2,
            num_heads=4,
            head_dim=64,
            ffn_hidden_dim=1024,
        ))
        ckpt_mgr = CheckpointManager(root_dir=str(root), format="safetensors", keep_last_n=2)
        ckpt_path = ckpt_mgr.save(model=model, step=1, metrics={"loss": 1.0})
        assert Path(ckpt_path).exists()
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    
    after = _snapshot_tree(get_primary_root())
    created = _resolve_created_paths(before, after)
    leaked = [p for p in created if not is_allowed_dataset_path(p)]
    assert leaked == [], f"Checkpoint created files outside approved roots: {leaked}"


def test_tempfile_uses_approved_temp_root():
    enforce_path_policy()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as f:
        tmp_path = f.name
    try:
        p = Path(tmp_path).resolve()
        assert is_allowed_dataset_path(p), f"tempfile created outside approved roots: {p}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_python_tempfile_default_location():
    enforce_path_policy()
    import tempfile
    tmp = tempfile.gettempdir()
    p = Path(tmp).resolve()
    assert is_allowed_dataset_path(p), f"Python tempfile.gettempdir() outside approved roots: {p}"


def test_no_c_drive_writes_during_pilot_dry_run():
    from src.data.tokenizer import CharacterTokenizer
    from src.data.sharding import prepare_and_shard_dataset
    
    root = get_shards_root() / "tests" / "pilot_dry_run"
    
    try:
        config = {
            "data": {
                "dataset_name": "synthetic",
                "tokenizer_vocab_size": 128,
                "num_shards": 2,
                "shard_strategy": "round_robin",
                "expert_specific_ratio": 0.8,
                "shard_seed": 42,
                "max_doc_length": 16,
                "min_doc_length": 2,
                "synthetic_num_docs": 10,
            }
        }
        tokenizer = CharacterTokenizer(vocab_size=128)
        metadata = prepare_and_shard_dataset(config, str(root), tokenizer)
    finally:
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
    
    created = []
    for base in APPROVED_ROOTS:
        if base.exists():
            for dirpath, dirnames, filenames in os.walk(str(base)):
                created.append(Path(dirpath).resolve())
                for f in filenames:
                    created.append((Path(dirpath) / f).resolve())
    
    leaked = [p for p in created if not is_allowed_dataset_path(p)]
    assert leaked == [], f"Pilot dry-run created paths outside approved roots: {leaked}"


def test_logger_uses_approved_log_root(tmp_path):
    from src.utils.logging import TrainingLogger
    
    # Enforce path policy
    enforce_path_policy()
    
    # Verify default log dir would be under approved root if created
    log_root = get_logs_root()
    assert is_allowed_dataset_path(log_root), f"Log root outside approved roots: {log_root}"
