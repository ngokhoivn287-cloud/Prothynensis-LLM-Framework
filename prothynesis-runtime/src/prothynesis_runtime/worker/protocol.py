from __future__ import annotations

import enum
import time
from typing import Any


class WorkerState(enum.Enum):
    IDLE = "idle"
    REGISTERED = "registered"
    CLAIMED = "claimed"
    RUNNING = "running"
    VALIDATING = "validating"
    SUBMITTING = "submitting"
    PAUSED = "paused"
    ERROR = "error"


class WorkerProtocol:
    def __init__(self, worker_id: str, runtime_version: str = "0.1.0") -> None:
        self.worker_id = worker_id
        self.runtime_version = runtime_version
        self._state = WorkerState.IDLE
        self._hardware_report: dict[str, Any] | None = None

    def register(self, hardware_report: dict[str, Any]) -> dict[str, Any]:
        self._hardware_report = hardware_report
        self._state = WorkerState.REGISTERED
        return {
            "worker_id": self.worker_id,
            "runtime_version": self.runtime_version,
            "hardware": hardware_report,
            "timestamp": time.time(),
        }

    def heartbeat(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self._state.value,
            "timestamp": time.time(),
            "vram": self._hardware_report.get("vram_gb") if self._hardware_report else None,
            "ram": self._hardware_report.get("ram_gb") if self._hardware_report else None,
            "temperature": self._hardware_report.get("temperature_c") if self._hardware_report else None,
        }

    def claim_job(self, job_spec: dict[str, Any]) -> dict[str, Any]:
        self._state = WorkerState.CLAIMED
        return {
            "worker_id": self.worker_id,
            "job_id": job_spec.get("job_id"),
            "status": "claimed",
            "timestamp": time.time(),
        }

    def submit_result(self, package_path: str, validation_result: dict[str, Any]) -> dict[str, Any]:
        self._state = WorkerState.SUBMITTING
        payload = {
            "worker_id": self.worker_id,
            "package_path": package_path,
            "validation_result": validation_result,
            "timestamp": time.time(),
        }
        self._state = WorkerState.IDLE
        return payload

    def get_state(self) -> WorkerState:
        return self._state

    def set_state(self, state: WorkerState) -> None:
        self._state = state

    def report_progress(
        self,
        step: int,
        total: int,
        loss: float | None = None,
        tokens: int | None = None,
    ) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "state": self._state.value,
            "progress": {
                "step": step,
                "total": total,
                "loss": loss,
                "tokens": tokens,
            },
            "timestamp": time.time(),
        }
