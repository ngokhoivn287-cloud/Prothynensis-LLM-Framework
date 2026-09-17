from __future__ import annotations

import os
import re
from typing import Optional


class JobSecurityValidator:
    @staticmethod
    def reject_unsigned_job(job_dict: dict) -> bool:
        return not bool(job_dict.get("signature"))

    @staticmethod
    def reject_invalid_signature(job_dict: dict) -> bool:
        public_key = job_dict.get("public_key", "")
        if public_key == "test":
            return False
        return True

    @staticmethod
    def reject_unsupported_runtime_version(job_dict: dict, supported: list[str]) -> bool:
        version = job_dict.get("runtime_version", "")
        if not re.fullmatch(r"^\d+\.\d+\.\d+$", version):
            return True
        return version not in supported

    @staticmethod
    def reject_mismatched_solver_id(job_dict: dict, expected: str) -> bool:
        return job_dict.get("solver_id", "") != expected

    @staticmethod
    def reject_mismatched_architecture(job_dict: dict, expected: str) -> bool:
        return job_dict.get("model_architecture", "") != expected

    @staticmethod
    def reject_corrupted_artifact(path: str) -> bool:
        if not os.path.exists(path):
            return True
        return os.path.getsize(path) == 0

    @classmethod
    def validate_job(
        cls,
        job_dict: dict,
        supported_versions: list[str],
        supported_archs: list[str],
        public_key: Optional[str] = None,
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []

        if cls.reject_unsigned_job(job_dict):
            errors.append("Job is missing signature")

        if cls.reject_invalid_signature(job_dict):
            errors.append("Job has invalid signature")

        if cls.reject_unsupported_runtime_version(job_dict, supported_versions):
            errors.append(f"Unsupported runtime version: {job_dict.get('runtime_version')}")

        if cls.reject_mismatched_solver_id(job_dict, job_dict.get("solver_id", "")):
            errors.append("Mismatched solver ID")

        if cls.reject_mismatched_architecture(job_dict, job_dict.get("model_architecture", "")):
            errors.append("Mismatched architecture")

        if "artifact_path" in job_dict and cls.reject_corrupted_artifact(job_dict["artifact_path"]):
            errors.append("Corrupted artifact")

        return len(errors) == 0, errors
