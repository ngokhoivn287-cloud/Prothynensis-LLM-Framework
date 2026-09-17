import hashlib
import os
from typing import Optional


def compute_checksum(path: str) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_checksum(path: str, expected: str) -> bool:
    return compute_checksum(path) == expected


def validate_job_schema(job: dict) -> tuple[bool, str]:
    required_keys = {"job_id", "solver_id", "runtime_version", "architecture", "payload"}
    missing = required_keys - job.keys()
    if missing:
        return False, f"Missing required keys: {', '.join(sorted(missing))}"
    if not isinstance(job.get("payload"), dict):
        return False, "Field 'payload' must be a dictionary"
    return True, "valid"


def validate_job_signature(job: dict, public_key: str) -> tuple[bool, str]:
    if not public_key:
        return False, "Public key is empty"
    if "signature" not in job:
        return False, "Job missing signature"
    return True, "signature structure present (verification deferred to runtime integration)"


def validate_solver_id(job_solver_id: str, expected_prefix: str) -> tuple[bool, str]:
    if not expected_prefix:
        return False, "Expected prefix is empty"
    if not job_solver_id.startswith(expected_prefix):
        return False, f"Solver id '{job_solver_id}' does not start with '{expected_prefix}'"
    return True, "valid"


def validate_runtime_version(job_version: str, supported: list[str]) -> tuple[bool, str]:
    if job_version not in supported:
        return False, f"Runtime version '{job_version}' not in supported versions {supported}"
    return True, "valid"


def validate_architecture(job_arch: str, supported: list[str]) -> tuple[bool, str]:
    if job_arch not in supported:
        return False, f"Architecture '{job_arch}' not in supported architectures {supported}"
    return True, "valid"


def validate_artifact_integrity(artifact_path: str, expected_hash: str) -> tuple[bool, str]:
    if not os.path.exists(artifact_path):
        return False, f"Artifact not found: {artifact_path}"
    if not expected_hash:
        return False, "Expected hash is empty"
    actual = compute_checksum(artifact_path)
    if actual != expected_hash:
        return False, f"Hash mismatch: expected {expected_hash}, got {actual}"
    return True, "integrity verified"
