import json

import pytest
from pydantic import ValidationError

from prothynesis_runtime.scheduler.job_spec import JobSpec, JobValidator
from prothynesis_runtime.validation.job_validator import JobSecurityValidator
from prothynesis_runtime.utils.helpers import canonical_json


def test_job_spec_valid():
    job = JobSpec(
        job_id="job_2024_0000001",
        solver_id="1234567",
        runtime_version="0.1.0",
        model_architecture="cpu",
        seed=42,
        dataset_id="ds_001",
        token_budget=10_000_000,
        sequence_length=1024,
        training_config_hash="abc",
        validation_config_hash="def",
        job_hash="ghi",
        signature="sig",
        created_at="2024-01-01T00:00:00",
    )
    assert job.job_id == "job_2024_0000001"
    assert job.solver_id == "1234567"
    assert job.runtime_version == "0.1.0"


def test_job_spec_invalid_solver_id():
    with pytest.raises(ValidationError):
        JobSpec(
            job_id="job_2024_0000001",
            solver_id="bad_id",
            runtime_version="0.1.0",
            model_architecture="cpu",
            seed=42,
            dataset_id="ds_001",
            token_budget=10_000_000,
            sequence_length=1024,
            training_config_hash="abc",
            validation_config_hash="def",
            job_hash="ghi",
            signature="sig",
            created_at="2024-01-01T00:00:00",
        )


def test_job_spec_invalid_runtime_version():
    with pytest.raises(ValidationError):
        JobSpec(
            job_id="job_2024_0000001",
            solver_id="1234567",
            runtime_version="bad",
            model_architecture="cpu",
            seed=42,
            dataset_id="ds_001",
            token_budget=10_000_000,
            sequence_length=1024,
            training_config_hash="abc",
            validation_config_hash="def",
            job_hash="ghi",
            signature="sig",
            created_at="2024-01-01T00:00:00",
        )


def test_job_spec_invalid_token_budget():
    with pytest.raises(ValidationError):
        JobSpec(
            job_id="job_2024_0000001",
            solver_id="1234567",
            runtime_version="0.1.0",
            model_architecture="cpu",
            seed=42,
            dataset_id="ds_001",
            token_budget=-1,
            sequence_length=1024,
            training_config_hash="abc",
            validation_config_hash="def",
            job_hash="ghi",
            signature="sig",
            created_at="2024-01-01T00:00:00",
        )


def test_job_validator_validate_job_schema_valid():
    job_dict = {
        "job_id": "job_2024_0000001",
        "solver_id": "1234567",
        "runtime_version": "0.1.0",
        "model_architecture": "cpu",
        "seed": 42,
        "dataset_id": "ds_001",
        "token_budget": 10_000_000,
        "sequence_length": 1024,
        "training_config_hash": "abc",
        "validation_config_hash": "def",
        "job_hash": "ghi",
        "signature": "sig",
        "created_at": "2024-01-01T00:00:00",
    }
    ok, msg, spec = JobValidator.validate_job_spec(job_dict)
    assert ok is True
    assert msg == "Valid job spec"
    assert spec is not None
    assert spec.job_id == "job_2024_0000001"


def test_job_validator_validate_job_schema_invalid():
    job_dict = {"job_id": "job_2024_0000001"}
    ok, msg, spec = JobValidator.validate_job_spec(job_dict)
    assert ok is False
    assert spec is None
    assert "validation error" in msg.lower() or "job_id" in msg.lower()


def test_job_validator_validate_token_budget_in_range():
    ok, msg = JobValidator.validate_token_budget(50_000_000)
    assert ok is True
    assert "within range" in msg


def test_job_validator_validate_token_budget_out_of_range():
    ok, msg = JobValidator.validate_token_budget(0)
    assert ok is False
    assert "out of range" in msg


def test_job_security_validator_reject_unsigned():
    job = {"solver_id": "1234567"}
    assert JobSecurityValidator.reject_unsigned_job(job) is True


def test_job_security_validator_reject_invalid_signature():
    job = {"signature": "sig", "public_key": "bad_key"}
    assert JobSecurityValidator.reject_invalid_signature(job) is True


def test_canonical_json_stable():
    obj = {"b": 2, "a": 1}
    first = canonical_json(obj)
    second = canonical_json(obj)
    assert first == second
    parsed = json.loads(first)
    assert parsed == {"a": 1, "b": 2}
