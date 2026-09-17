import hashlib
import os
import tempfile

import pytest

from prothynesis_runtime.security.validator import (
    compute_checksum,
    verify_checksum,
    validate_architecture,
    validate_artifact_integrity,
    validate_job_schema,
    validate_job_signature,
    validate_runtime_version,
    validate_solver_id,
)


def test_compute_checksum_consistent():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"consistent data")
        tmp_path = tmp.name
    try:
        first = compute_checksum(tmp_path)
        second = compute_checksum(tmp_path)
        assert first == second
        assert len(first) == 64
    finally:
        os.unlink(tmp_path)


def test_verify_checksum_valid():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello world")
        tmp_path = tmp.name
    try:
        expected = compute_checksum(tmp_path)
        assert verify_checksum(tmp_path, expected) is True
    finally:
        os.unlink(tmp_path)


def test_verify_checksum_invalid():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello world")
        tmp_path = tmp.name
    try:
        assert verify_checksum(tmp_path, "bad_hash") is False
    finally:
        os.unlink(tmp_path)


def test_validate_job_schema_valid():
    job = {
        "job_id": "job_2024_0000001",
        "solver_id": "1234567",
        "runtime_version": "0.1.0",
        "architecture": "cpu",
        "payload": {"key": "value"},
    }
    ok, msg = validate_job_schema(job)
    assert ok is True
    assert msg == "valid"


def test_validate_job_schema_missing_fields():
    job = {"job_id": "job_2024_0000001"}
    ok, msg = validate_job_schema(job)
    assert ok is False
    assert "Missing required keys" in msg


def test_validate_job_signature_valid_test_key():
    job = {"signature": "abc123"}
    ok, msg = validate_job_signature(job, "test")
    assert ok is True
    assert "signature structure present" in msg


def test_validate_job_signature_invalid():
    job = {}
    ok, msg = validate_job_signature(job, "test")
    assert ok is False
    assert msg == "Job missing signature"


def test_validate_solver_id_valid():
    ok, msg = validate_solver_id("solver-abc", "solver-")
    assert ok is True
    assert msg == "valid"


def test_validate_solver_id_invalid():
    ok, msg = validate_solver_id("other-abc", "solver-")
    assert ok is False
    assert "does not start with" in msg


def test_validate_runtime_version_supported():
    ok, msg = validate_runtime_version("0.1.0", ["0.1.0", "0.2.0"])
    assert ok is True
    assert msg == "valid"


def test_validate_runtime_version_unsupported():
    ok, msg = validate_runtime_version("9.9.9", ["0.1.0", "0.2.0"])
    assert ok is False
    assert "not in supported versions" in msg


def test_validate_architecture_supported():
    ok, msg = validate_architecture("cpu", ["cpu", "cuda"])
    assert ok is True
    assert msg == "valid"


def test_validate_architecture_unsupported():
    ok, msg = validate_architecture("tpu", ["cpu", "cuda"])
    assert ok is False
    assert "not in supported architectures" in msg


def test_validate_artifact_integrity_valid():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello world")
        tmp_path = tmp.name
    try:
        expected = compute_checksum(tmp_path)
        ok, msg = validate_artifact_integrity(tmp_path, expected)
        assert ok is True
        assert msg == "integrity verified"
    finally:
        os.unlink(tmp_path)


def test_validate_artifact_integrity_missing():
    ok, msg = validate_artifact_integrity("/nonexistent/path", "abc")
    assert ok is False
    assert "Artifact not found" in msg
