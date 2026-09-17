from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import requests
import torch
from pydantic import BaseModel


class ValidationResult(BaseModel):
    job_id: str
    passed: bool
    score: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None


class JobSpec(BaseModel):
    job_id: str
    dataset_url: str
    model_config: dict[str, Any] = field(default_factory=dict)
    training_steps: int = 100
    output_format: str = "pt"


@dataclass
class HardwareLimits:
    max_gpu_utilization: float = 0.9
    max_vram_usage_gb: float | None = None
    max_temperature_c: float | None = None
    max_concurrent_solvers: int = 1
    pause_while_user_active: bool = False


class WorkerClient:
    def __init__(
        self,
        coordinator_url: str,
        worker_id: str,
        runtime_version: str,
        artifact_store: Path,
    ) -> None:
        self.coordinator_url = coordinator_url.rstrip("/")
        self.worker_id = worker_id
        self.runtime_version = runtime_version
        self.artifact_store = Path(artifact_store)
        self.artifact_store.mkdir(parents=True, exist_ok=True)
        self._paused = False
        self._stopped = False

    def _get_hardware_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        }
        if torch.cuda.is_available():
            info.update({
                "cuda_available": True,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            })
        else:
            info.update({
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_device_name": None,
                "vram_gb": 0.0,
            })
        return info

    def register(self) -> bool:
        hardware_report = self._get_hardware_info()
        payload = {
            "worker_id": self.worker_id,
            "runtime_version": self.runtime_version,
            "hardware": hardware_report,
        }
        try:
            resp = requests.post(
                f"{self.coordinator_url}/workers/register",
                json=payload,
                timeout=30,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def claim_job(self) -> dict[str, Any] | None:
        try:
            resp = requests.post(
                f"{self.coordinator_url}/workers/{self.worker_id}/claim",
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                return JobSpec(**data).model_dump() if data.get("job_id") else None
            return None
        except requests.RequestException:
            return None

    def download_job_data(self, job_spec: dict[str, Any]) -> Path:
        dataset_url = job_spec["dataset_url"]
        job_id = job_spec["job_id"]
        local_path = self.artifact_store / f"{job_id}_dataset.pt"
        if not local_path.exists():
            resp = requests.get(dataset_url, timeout=120)
            resp.raise_for_status()
            local_path.write_bytes(resp.content)
        return local_path

    def train_solver(
        self,
        job_spec: dict[str, Any],
        dataset_path: Path,
        output_dir: Path,
        hardware_limits: HardwareLimits,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        job = JobSpec(**job_spec)
        training_steps = job.training_steps
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()
        dummy_x = torch.randn(32, 10)
        dummy_y = torch.randn(32, 1)
        losses: list[float] = []
        tokens_seen = 0
        for step in range(training_steps):
            if self._stopped:
                break
            while self._paused:
                time.sleep(1)
            model.train()
            optimizer.zero_grad()
            pred = model(dummy_x)
            loss = criterion(pred, dummy_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            tokens_seen += dummy_x.shape[0]
            if torch.cuda.is_available():
                util = torch.cuda.utilization()
                vram_gb = torch.cuda.memory_allocated() / (1024**3)
                if hardware_limits.max_vram_usage_gb and vram_gb > hardware_limits.max_vram_usage_gb:
                    torch.cuda.empty_cache()
        torch.save(model.state_dict(), output_dir / "model.pt")
        metadata = {
            "job_id": job.job_id,
            "training_steps": training_steps,
            "final_loss": losses[-1] if losses else None,
            "tokens_seen": tokens_seen,
            "loss_history": losses,
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    def validate_result(self, output_dir: Path, job_spec: dict[str, Any]) -> ValidationResult:
        output_dir = Path(output_dir)
        model_path = output_dir / "model.pt"
        metadata_path = output_dir / "metadata.json"
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        if not model_path.exists():
            return ValidationResult(
                job_id=job_spec["job_id"],
                passed=False,
                error_message="Model output not found",
            )
        try:
            model = torch.nn.Linear(10, 1)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.eval()
            with torch.no_grad():
                dummy_x = torch.randn(4, 10)
                pred = model(dummy_x)
                score = float(pred.abs().mean().item())
            passed = score < 10.0 and metadata.get("final_loss") is not None
            return ValidationResult(
                job_id=job_spec["job_id"],
                passed=passed,
                score=score,
                metrics={"final_loss": metadata.get("final_loss"), "validation_score": score},
            )
        except Exception as e:
            return ValidationResult(
                job_id=job_spec["job_id"],
                passed=False,
                error_message=str(e),
            )

    def package_result(
        self,
        output_dir: Path,
        metadata: dict[str, Any],
        validation_result: ValidationResult,
        job_spec: dict[str, Any],
    ) -> Path:
        output_dir = Path(output_dir)
        package_path = self.artifact_store / f"{job_spec['job_id']}_result.zip"
        import zipfile
        with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file_name in ["model.pt", "metadata.json"]:
                file_path = output_dir / file_name
                if file_path.exists():
                    zf.write(file_path, arcname=file_name)
            zf.writestr("validation.json", validation_result.model_dump_json())
            zf.writestr("package_metadata.json", json.dumps(metadata))
        return package_path

    def submit_result(
        self,
        package_path: Path,
        job_spec: dict[str, Any],
        validation_result: ValidationResult,
    ) -> bool:
        package_path = Path(package_path)
        if not package_path.exists():
            return False
        try:
            with open(package_path, "rb") as f:
                files = {"package": f}
                data = {
                    "job_id": job_spec["job_id"],
                    "worker_id": self.worker_id,
                    "validation_result": validation_result.model_dump_json(),
                }
                resp = requests.post(
                    f"{self.coordinator_url}/jobs/{job_spec['job_id']}/submit",
                    files=files,
                    data=data,
                    timeout=120,
                )
                return resp.status_code == 200
        except requests.RequestException:
            return False

    def run_job(self, job_spec: dict[str, Any], hardware_limits: HardwareLimits) -> dict[str, Any]:
        job_id = job_spec["job_id"]
        dataset_path = self.download_job_data(job_spec)
        output_dir = self.artifact_store / f"job_{job_id}"
        metadata = self.train_solver(job_spec, dataset_path, output_dir, hardware_limits)
        validation_result = self.validate_result(output_dir, job_spec)
        package_path = self.package_result(output_dir, metadata, validation_result, job_spec)
        success = self.submit_result(package_path, job_spec, validation_result)
        return {
            "job_id": job_id,
            "success": success,
            "validation_passed": validation_result.passed,
            "package_path": str(package_path),
        }

    def pause(self) -> None:
        self._paused = True

    def stop(self) -> None:
        self._stopped = True

    def get_status(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "runtime_version": self.runtime_version,
            "paused": self._paused,
            "stopped": self._stopped,
            "artifact_store": str(self.artifact_store),
            "hardware": self._get_hardware_info(),
        }
