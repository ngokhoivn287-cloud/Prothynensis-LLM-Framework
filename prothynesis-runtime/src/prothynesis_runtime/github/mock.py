from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from prothynesis_runtime.github.oauth import GitHubAuthResult, GitHubProfile


class MockGitHubClient:
    def __init__(self, mock_profiles: dict[str, GitHubProfile] | None = None) -> None:
        self.mock_profiles = mock_profiles or {
            "default": GitHubProfile(
                id=12345,
                login="mockuser",
                name="Mock User",
                email="mock@example.com",
                avatar_url="https://example.com/avatar.png",
            )
        }
        self._tokens: dict[str, GitHubAuthResult] = {}

    def authorize(self, state: str | None = None) -> str:
        return f"https://github.com/login/oauth/authorize?client_id=mock&state={state or ''}"

    def exchange_code(self, code: str) -> GitHubAuthResult:
        username = "mockuser"
        profile = self.mock_profiles.get(username)
        result = GitHubAuthResult(
            access_token=f"mock_token_{code}",
            token_type="bearer",
            scope="read:user",
            github_id=profile.id if profile else 12345,
            username=username,
            expires_at=time.time() + 3600,
        )
        self._tokens[result.access_token] = result
        return result

    def get_user_profile(self, token: str) -> GitHubProfile | None:
        result = self._tokens.get(token)
        if not result or not result.username:
            return None
        return self.mock_profiles.get(result.username)

    def revoke_token(self, token: str) -> bool:
        if token in self._tokens:
            del self._tokens[token]
            return True
        return False


class MockCoordinatorClient:
    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url
        self._jobs: dict[str, dict[str, Any]] = {
            "job_001": {
                "job_id": "job_001",
                "dataset_url": "mock://dataset/job_001",
                "training_steps": 100,
                "status": "pending",
            },
            "job_002": {
                "job_id": "job_002",
                "dataset_url": "mock://dataset/job_002",
                "training_steps": 200,
                "status": "pending",
            },
        }
        self._results: dict[str, list[dict[str, Any]]] = {}

    def register_worker(self, worker_payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "registered",
            "worker_id": worker_payload.get("worker_id", "unknown"),
            "timestamp": time.time(),
        }

    def get_available_jobs(self) -> list[dict[str, Any]]:
        return [
            job for job in self._jobs.values() if job.get("status") == "pending"
        ]

    def claim_job(self, job_id: str, worker_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "Job not found"}
        if job.get("status") != "pending":
            return {"status": "error", "error": "Job already claimed"}
        job["status"] = "claimed"
        job["claimed_by"] = worker_id
        job["claimed_at"] = time.time()
        return {"status": "claimed", "job_id": job_id, "worker_id": worker_id}

    def submit_result(
        self,
        job_id: str,
        worker_id: str,
        package_metadata: dict[str, Any],
        validation_result: Any,
    ) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "error", "error": "Job not found"}
        job["status"] = "completed"
        job["completed_by"] = worker_id
        job["completed_at"] = time.time()
        result_entry = {
            "job_id": job_id,
            "worker_id": worker_id,
            "package_metadata": package_metadata,
            "validation_result": validation_result.model_dump() if hasattr(validation_result, "model_dump") else str(validation_result),
            "submitted_at": time.time(),
        }
        self._results.setdefault(job_id, []).append(result_entry)
        return {"status": "submitted", "job_id": job_id}

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "not_found"}
        return {
            "job_id": job_id,
            "status": job.get("status", "unknown"),
            "claimed_by": job.get("claimed_by"),
            "completed_by": job.get("completed_by"),
        }


def mock_train_solver(
    job_spec: dict[str, Any],
    dataset_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_steps = job_spec.get("training_steps", 100)
    losses = [max(0.1, 1.0 - (i / training_steps) + random.uniform(-0.05, 0.05)) for i in range(training_steps)]
    metadata = {
        "job_id": job_spec.get("job_id", "unknown"),
        "training_steps": training_steps,
        "final_loss": losses[-1] if losses else None,
        "tokens_seen": training_steps * 32,
        "loss_history": losses,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata
