import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PackageResult:
    package_path: str
    checkpoint_path: str
    checksum: str
    metadata: Dict[str, Any]


def compute_package_checksum(package_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(package_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def package_solver_result(
    checkpoint_path: str,
    metadata: Dict[str, Any],
    validation_result: Dict[str, Any],
    job_id: str,
    solver_id: str,
    runtime_version: str,
) -> Path:
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = checkpoint.parent / f"solver_{solver_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_target = checkpoint_dir / checkpoint.name
    checkpoint_target.write_bytes(checkpoint.read_bytes())

    checksum = _sha256_file(checkpoint_target)

    manifest = {
        "solver_id": solver_id,
        "job_id": job_id,
        "runtime_version": runtime_version,
        "training_metadata": metadata.get("training_metadata", {}),
        "contributor_metadata": metadata.get("contributor_metadata", {}),
        "checkpoint_hash": checksum,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    validation = {
        "validation_status": validation_result.get("status", "unknown"),
        "checks": validation_result.get("checks", {}),
        "timestamp": validation_result.get("timestamp", ""),
    }
    (output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    (output_dir / "checksum.sha256").write_text(checksum, encoding="utf-8")

    return output_dir


def validate_package(package_path: str) -> tuple[bool, str]:
    root = Path(package_path)
    if not root.exists():
        return False, f"Package path does not exist: {package_path}"
    manifest_path = root / "manifest.json"
    validation_path = root / "validation.json"
    checksum_path = root / "checksum.sha256"
    if not manifest_path.exists():
        return False, "manifest.json missing"
    if not validation_path.exists():
        return False, "validation.json missing"
    if not checksum_path.exists():
        return False, "checksum.sha256 missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        json.loads(validation_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"JSON parse error: {e}"
    checkpoint_dir = root / "checkpoint"
    if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        return False, "checkpoint directory missing or empty"
    return True, "package valid"


def _sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()
