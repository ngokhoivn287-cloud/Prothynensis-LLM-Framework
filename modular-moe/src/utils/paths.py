"""Centralized filesystem path validator for Prothynesis.

Approved roots are configurable via PROTHYNESIS_DATA_ROOT environment variable.
Default: D:\\NgoPROJECT\\ProthynensisDatasets (Windows) or /mnt/d/NgoPROJECT/ProthynensisDatasets (WSL)

All dataset, training, preprocessing, download, cache, checkpoint,
log, manifest, shard, and artifact paths MUST resolve within these roots.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional, Union


def _get_default_roots() -> list[Path]:
    """Return default approved roots based on platform."""
    if sys.platform == "linux":
        return [Path("/mnt/d/NgoPROJECT/ProthynensisDatasets"), Path("/mnt/d/NgoPROJECT/ProthynensisDatasets/temp")]
    return [Path(r"D:\NgoPROJECT\ProthynensisDatasets"), Path(r"D:\NgoPROJECT\ProthynensisDatasets\temp")]


def _get_approved_roots() -> list[Path]:
    """Return approved roots from environment variable or defaults."""
    env_root = os.environ.get("PROTHYNESIS_DATA_ROOT")
    if env_root:
        primary = Path(env_root)
        return [primary, primary / "temp"]
    return _get_default_roots()

_RESOLVED_APPROVED: Optional[Path] = None


def _ensure_roots() -> Path:
    """Ensure approved roots exist and return primary root."""
    roots = _get_approved_roots()
    primary = roots[0]
    temp_root = roots[1] if len(roots) > 1 else primary / "temp"
    primary.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)
    return primary


def is_allowed_dataset_path(path: Union[str, Path]) -> bool:
    """Check whether *path* resolves inside the approved dataset roots."""
    global _RESOLVED_APPROVED
    roots = _get_approved_roots()
    primary = roots[0]
    temp_root = roots[1] if len(roots) > 1 else primary / "temp"
    try:
        resolved = Path(path).resolve()
    except (OSError, RuntimeError):
        return False
    if resolved == primary or resolved == temp_root:
        return True
    try:
        resolved.relative_to(primary)
        return True
    except ValueError:
        pass
    try:
        resolved.relative_to(temp_root)
        return True
    except ValueError:
        pass
    return False


def assert_allowed_path(path: Union[str, Path], context: str = "") -> Path:
    """Validate and return resolved path; raise if outside approved roots."""
    resolved = Path(path).resolve()
    if not is_allowed_dataset_path(resolved):
        raise PermissionError(
            f"Refusing to use path outside approved dataset roots: {resolved}"
            + (f" ({context})" if context else "")
        )
    return resolved


def get_primary_root() -> Path:
    """Return the primary approved dataset root, creating it if needed."""
    return _ensure_roots()


def get_temp_root() -> Path:
    """Return the approved temporary root under the primary dataset root."""
    primary = _ensure_roots()
    temp_root = primary / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return temp_root


def get_checkpoints_root() -> Path:
    """Return approved checkpoint root."""
    return get_primary_root() / "checkpoints"


def get_shards_root() -> Path:
    """Return approved shard root."""
    return get_primary_root() / "shards"


def get_logs_root() -> Path:
    """Return approved logs root."""
    return get_primary_root() / "logs"


def get_manifests_root() -> Path:
    """Return approved manifests root."""
    return get_primary_root() / "manifests"


def get_registry_root() -> Path:
    """Return approved model-registry root."""
    return get_primary_root() / "registry"


def enforce_path_policy() -> None:
    """Enforce environment-level path policy for the current process.

    Overrides Python tempdir, Hugging Face cache, and Torch Hub
    directories to stay within approved roots.
    """
    primary = get_primary_root()
    temp_root = get_temp_root()

    os.environ["TMPDIR"] = str(temp_root)
    os.environ["TEMP"] = str(temp_root)
    os.environ["TMP"] = str(temp_root)
    os.environ["XDG_CACHE_HOME"] = str(primary / ".cache")
    os.environ["HF_HOME"] = str(primary / ".cache" / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(primary / ".cache" / "huggingface" / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(primary / ".cache" / "huggingface" / "datasets")
    os.environ["TOKENIZERS_CACHE"] = str(primary / ".cache" / "tokenizers")
    os.environ["TORCH_HOME"] = str(primary / ".cache" / "torch")
    os.environ["PYTORCH_HOME"] = str(primary / ".cache" / "torch")
    os.environ["WANDB_DIR"] = str(primary / "wandb")

    import tempfile
    tempfile.tempdir = str(temp_root)
