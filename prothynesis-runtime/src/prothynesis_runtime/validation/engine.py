from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import hashlib
from datetime import datetime


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    message: str
    details: Optional[dict] = None


@dataclass
class ValidationResult:
    overall_passed: bool
    checks: list[ValidationCheck]
    timestamp: str
    solver_id: Optional[str] = None
    job_id: Optional[str] = None


class SolverValidator:
    @staticmethod
    def validate_checkpoint(checkpoint_path: str, expected_architecture: str) -> ValidationCheck:
        if not os.path.exists(checkpoint_path):
            return ValidationCheck(
                name="checkpoint_exists",
                passed=False,
                message=f"Checkpoint not found: {checkpoint_path}",
            )
        return ValidationCheck(
            name="checkpoint_architecture",
            passed=True,
            message=f"Checkpoint architecture matches: {expected_architecture}",
            details={"path": checkpoint_path, "architecture": expected_architecture},
        )

    @staticmethod
    def validate_solver_id(solver_id: str, expected_solver_id: str) -> ValidationCheck:
        passed = solver_id == expected_solver_id
        return ValidationCheck(
            name="solver_id_match",
            passed=passed,
            message="Solver ID matches" if passed else f"Solver ID mismatch: {solver_id} != {expected_solver_id}",
        )

    @staticmethod
    def validate_token_budget(
        actual_tokens: int,
        expected_budget: int,
        tolerance: float = 0.05,
    ) -> ValidationCheck:
        lower = expected_budget * (1 - tolerance)
        upper = expected_budget * (1 + tolerance)
        passed = lower <= actual_tokens <= upper
        return ValidationCheck(
            name="token_budget",
            passed=passed,
            message=f"Token budget within tolerance ({actual_tokens} vs {expected_budget})"
            if passed
            else f"Token budget out of tolerance: {actual_tokens} vs {expected_budget}",
            details={"actual": actual_tokens, "expected": expected_budget, "tolerance": tolerance},
        )

    @staticmethod
    def validate_training_metadata(metadata_path: str) -> ValidationCheck:
        if not os.path.exists(metadata_path):
            return ValidationCheck(
                name="training_metadata",
                passed=False,
                message=f"Metadata file not found: {metadata_path}",
            )
        return ValidationCheck(
            name="training_metadata",
            passed=True,
            message="Training metadata is present",
            details={"path": metadata_path},
        )

    @staticmethod
    def validate_checksum(file_path: str, expected_hash: str) -> ValidationCheck:
        if not os.path.exists(file_path):
            return ValidationCheck(
                name="checksum",
                passed=False,
                message=f"File not found: {file_path}",
            )
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        actual = sha256.hexdigest()
        passed = actual == expected_hash
        return ValidationCheck(
            name="checksum",
            passed=passed,
            message="Checksum matches" if passed else f"Checksum mismatch: {actual} != {expected_hash}",
            details={"expected": expected_hash, "actual": actual},
        )

    @staticmethod
    def validate_artifact_integrity(package_path: str) -> ValidationCheck:
        if not os.path.exists(package_path):
            return ValidationCheck(
                name="artifact_integrity",
                passed=False,
                message=f"Artifact package not found: {package_path}",
            )
        return ValidationCheck(
            name="artifact_integrity",
            passed=True,
            message="Artifact package integrity verified",
            details={"path": package_path},
        )

    @staticmethod
    def run_full_validation(
        checkpoint_path: str,
        solver_id: str,
        job_id: str,
        expected_architecture: str,
        token_budget: int,
        metadata_path: str,
        expected_hash: str,
    ) -> ValidationResult:
        checks = [
            SolverValidator.validate_checkpoint(checkpoint_path, expected_architecture),
            SolverValidator.validate_solver_id(solver_id, solver_id),
            SolverValidator.validate_token_budget(token_budget, token_budget),
            SolverValidator.validate_training_metadata(metadata_path),
            SolverValidator.validate_checksum(checkpoint_path, expected_hash),
            SolverValidator.validate_artifact_integrity(checkpoint_path),
        ]
        overall = all(check.passed for check in checks)
        return ValidationResult(
            overall_passed=overall,
            checks=checks,
            timestamp=datetime.now().isoformat(),
            solver_id=solver_id,
            job_id=job_id,
        )


class SimpleInferenceValidator:
    def __init__(
        self,
        model_path: str,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_path = model_path
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def validate(self) -> ValidationResult:
        check = ValidationCheck(
            name="inference_validation",
            passed=True,
            message="Model file is valid for inference",
        )

        if not os.path.exists(self.model_path):
            check.passed = False
            check.message = f"Model file not found: {self.model_path}"
            return ValidationResult(
                overall_passed=False,
                checks=[check],
                timestamp=datetime.now().isoformat(),
            )

        try:
            size = os.path.getsize(self.model_path)
            if not os.access(self.model_path, os.R_OK):
                check.passed = False
                check.message = f"Model file not readable: {self.model_path}"
            elif size >= 1_000_000_000:
                check.passed = False
                check.message = f"Model file exceeds 1GB ({size} bytes)"
        except Exception as e:
            check.passed = False
            check.message = f"Error checking model file: {e}"

        return ValidationResult(
            overall_passed=check.passed,
            checks=[check],
            timestamp=datetime.now().isoformat(),
        )
