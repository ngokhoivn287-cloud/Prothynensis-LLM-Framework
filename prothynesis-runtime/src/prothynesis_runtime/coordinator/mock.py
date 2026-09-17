from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


class MockCoordinator:
    def __init__(self, storage_path: str | Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._workers: dict[str, dict[str, Any]] = {}
        self._jobs: dict[str, dict[str, Any]] = {}
        self._results: dict[str, dict[str, Any]] = {}
        self._population: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        data_file = self.storage_path / "data.json"
        if data_file.exists():
            try:
                with open(data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._workers = data.get("workers", {})
                self._jobs = data.get("jobs", {})
                self._results = data.get("results", {})
                self._population = data.get("population", [])
            except Exception:
                pass

    def _save(self) -> None:
        data_file = self.storage_path / "data.json"
        try:
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "workers": self._workers,
                        "jobs": self._jobs,
                        "results": self._results,
                        "population": self._population,
                    },
                    f,
                    indent=2,
                    default=str,
                )
        except Exception:
            pass

    def register_worker(self, worker_payload: dict[str, Any]) -> dict[str, Any]:
        worker_id = worker_payload.get("worker_id") or str(uuid.uuid4())
        record = {
            "worker_id": worker_id,
            "payload": worker_payload,
            "registered_at": time.time(),
            "status": "active",
        }
        self._workers[worker_id] = record
        self._save()
        return record

    def advertise_jobs(self) -> list[dict[str, Any]]:
        return [job for job in self._jobs.values() if job.get("status") == "pending"]

    def claim_job(self, job_id: str, worker_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "error", "reason": "job not found"}
        if job.get("status") != "pending":
            return {"status": "error", "reason": "job not available"}
        job["status"] = "claimed"
        job["claimed_by"] = worker_id
        job["claimed_at"] = time.time()
        self._save()
        return {"status": "claimed", "job_id": job_id, "worker_id": worker_id}

    def receive_result(
        self,
        job_id: str,
        worker_id: str,
        package_metadata: dict[str, Any],
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        result = {
            "job_id": job_id,
            "worker_id": worker_id,
            "package_metadata": package_metadata,
            "validation_result": validation_result,
            "received_at": time.time(),
        }
        self._results[job_id] = result
        job = self._jobs.get(job_id)
        if job:
            job["status"] = "completed"
        self._save()
        return {"status": "received", "job_id": job_id}

    def verify_result(self, job_id: str, package_path: str) -> dict[str, Any]:
        result = self._results.get(job_id)
        if not result:
            return {"status": "REJECTED", "reason": "no result found"}
        try:
            checksum = result.get("package_metadata", {}).get("checksum")
            if checksum:
                if not Path(package_path).exists():
                    return {"status": "REJECTED", "reason": "package not found"}
                sha = hashlib.sha256()
                with open(package_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        sha.update(chunk)
                if sha.hexdigest() != checksum:
                    return {"status": "REJECTED", "reason": "checksum mismatch"}
        except Exception as exc:
            return {"status": "REJECTED", "reason": str(exc)}
        return {"status": "ACCEPTED", "job_id": job_id}

    def get_population(self) -> list[dict[str, Any]]:
        return list(self._population)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"status": "unknown"}
        return job

    def list_jobs(self) -> list[dict[str, Any]]:
        return list(self._jobs.values())
