from __future__ import annotations

import re
import hashlib
import json
from datetime import datetime
from typing import Optional
from enum import Enum
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class JobStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUBMITTED = "SUBMITTED"
    VERIFYING = "VERIFYING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class JobSpec(BaseModel):
    model_config = ConfigDict(strict=True)

    job_id: str = Field(pattern=r"^job_\d{4}_\d{7}$")
    solver_id: str = Field(pattern=r"^\d{7}$")
    runtime_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    model_architecture: str
    seed: int = Field(ge=0, le=2**32 - 1)
    dataset_id: str
    token_budget: int = Field(gt=0)
    sequence_length: int = Field(gt=0)
    training_config_hash: str
    validation_config_hash: str
    job_hash: str
    signature: str
    created_at: str
    expires_at: Optional[str] = None

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v)
        except ValueError:
            raise ValueError("created_at must be a valid ISO datetime string")
        return v


class SolverTrainingConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    model_architecture: str
    seed: int
    dataset_id: str
    token_budget: int
    sequence_length: int
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    weight_decay: float = 0.01
    optimizer: str = "adamw"
    scheduler: str = "cosine"


class ValidationConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    validation_steps: int = 6
    validation_prompts: list[str] = []
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class JobRecord:
    job_id: str
    solver_id: str
    status: JobStatus
    assigned_worker: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    expires_at: Optional[str] = None


class JobValidator:
    @staticmethod
    def validate_job_spec(job_dict: dict) -> tuple[bool, str, Optional[JobSpec]]:
        try:
            spec = JobSpec(**job_dict)
            return True, "Valid job spec", spec
        except ValidationError as e:
            return False, str(e), None

    @staticmethod
    def validate_solver_id_format(solver_id: str) -> tuple[bool, str]:
        if re.fullmatch(r"^\d{7}$", solver_id):
            return True, "Valid solver ID format"
        return False, f"Invalid solver ID format: {solver_id}"

    @staticmethod
    def validate_runtime_version(version: str, supported: list[str]) -> tuple[bool, str]:
        if not re.fullmatch(r"^\d+\.\d+\.\d+$", version):
            return False, f"Invalid version format: {version}"
        if version in supported:
            return True, "Runtime version supported"
        return False, f"Unsupported runtime version: {version}"

    @staticmethod
    def validate_architecture(arch: str, supported: list[str]) -> tuple[bool, str]:
        if arch in supported:
            return True, "Architecture supported"
        return False, f"Unsupported architecture: {arch}"

    @staticmethod
    def validate_token_budget(
        budget: int,
        min_budget: int = 1_000_000,
        max_budget: int = 1_000_000_000,
    ) -> tuple[bool, str]:
        if min_budget <= budget <= max_budget:
            return True, "Token budget within range"
        return False, f"Token budget {budget} out of range [{min_budget}, {max_budget}]"

    @staticmethod
    def validate_job_hash(job_dict: dict, expected_hash: str) -> tuple[bool, str]:
        canonical = json.dumps(job_dict, sort_keys=True, separators=(",", ":"))
        actual_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if actual_hash == expected_hash:
            return True, "Job hash matches"
        return False, f"Job hash mismatch: expected {expected_hash}, got {actual_hash}"

    @staticmethod
    def validate_signature(job_dict: dict, signature: str, public_key: str) -> tuple[bool, str]:
        if public_key == "test":
            return True, "Signature valid (test mode)"
        return False, "signature validation requires coordinator public key"

    @staticmethod
    def validate_expiry(expires_at: Optional[str]) -> tuple[bool, str]:
        if expires_at is None:
            return True, "No expiry set"
        try:
            expiry = datetime.fromisoformat(expires_at)
            if expiry > datetime.utcnow():
                return True, "Job not expired"
            return False, "Job has expired"
        except ValueError:
            return False, f"Invalid expiry datetime format: {expires_at}"

    @staticmethod
    def is_job_supported(
        job_spec: JobSpec,
        supported_archs: list[str],
        supported_versions: list[str],
    ) -> tuple[bool, str]:
        valid_arch, msg = JobValidator.validate_architecture(job_spec.model_architecture, supported_archs)
        if not valid_arch:
            return False, msg
        valid_ver, msg = JobValidator.validate_runtime_version(job_spec.runtime_version, supported_versions)
        if not valid_ver:
            return False, msg
        return True, "Job is supported"
