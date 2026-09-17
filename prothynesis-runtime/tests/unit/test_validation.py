import os
import tempfile

import pytest

from prothynesis_runtime.validation.engine import (
    SolverValidator,
    ValidationCheck,
    ValidationResult,
)


def test_validation_check_creation():
    check = ValidationCheck(
        name="test_check",
        passed=True,
        message="All good",
        details={"key": "value"},
    )
    assert check.name == "test_check"
    assert check.passed is True
    assert check.message == "All good"
    assert check.details == {"key": "value"}


def test_validation_result_aggregation():
    checks = [
        ValidationCheck(name="c1", passed=True, message="ok"),
        ValidationCheck(name="c2", passed=False, message="fail"),
    ]
    result = ValidationResult(
        overall_passed=False,
        checks=checks,
        timestamp="2024-01-01T00:00:00",
        solver_id="solver_001",
        job_id="job_001",
    )
    assert result.overall_passed is False
    assert len(result.checks) == 2
    assert result.solver_id == "solver_001"
    assert result.job_id == "job_001"


def test_solver_validator_checkpoint_valid():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"checkpoint data")
        tmp_path = tmp.name
    try:
        check = SolverValidator.validate_checkpoint(tmp_path, "cpu")
        assert check.passed is True
        assert "matches" in check.message
        assert check.details is not None
    finally:
        os.unlink(tmp_path)


def test_solver_validator_checkpoint_missing():
    check = SolverValidator.validate_checkpoint("/nonexistent/path", "cpu")
    assert check.passed is False
    assert "not found" in check.message


def test_solver_validator_token_budget_within_tolerance():
    check = SolverValidator.validate_token_budget(1_050_000, 1_000_000, tolerance=0.05)
    assert check.passed is True
    assert "within tolerance" in check.message


def test_solver_validator_token_budget_outside_tolerance():
    check = SolverValidator.validate_token_budget(2_000_000, 1_000_000, tolerance=0.05)
    assert check.passed is False
    assert "out of tolerance" in check.message
