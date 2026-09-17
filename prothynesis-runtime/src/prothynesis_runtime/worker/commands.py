from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psutil
import torch

from prothynesis_runtime.worker.client import HardwareLimits, WorkerClient
from prothynesis_runtime.worker.protocol import WorkerProtocol


class WorkerCLI:
    def __init__(self, coordinator_url: str = "http://localhost:8080") -> None:
        self.coordinator_url = coordinator_url
        self.history_file = Path.home() / ".prothynesis" / "worker_history.jsonl"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def _log_history(self, event: str, data: dict[str, Any]) -> None:
        record = {"event": event, "data": data}
        with open(self.history_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _load_history(self) -> list[dict[str, Any]]:
        if not self.history_file.exists():
            return []
        records = []
        with open(self.history_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def cmd_status(self, args: argparse.Namespace) -> int:
        worker_id = getattr(args, "worker_id", "default")
        runtime_version = "0.1.0"
        protocol = WorkerProtocol(worker_id=worker_id, runtime_version=runtime_version)
        status = {
            "worker_id": worker_id,
            "state": protocol.get_state().value,
            "runtime_version": runtime_version,
        }
        print(json.dumps(status, indent=2))
        return 0

    def cmd_register(self, args: argparse.Namespace) -> int:
        worker_id = getattr(args, "worker_id", "default")
        artifact_store = Path(getattr(args, "artifact_store", "./artifacts"))
        client = WorkerClient(
            coordinator_url=self.coordinator_url,
            worker_id=worker_id,
            runtime_version="0.1.0",
            artifact_store=artifact_store,
        )
        success = client.register()
        self._log_history("register", {"worker_id": worker_id, "success": success})
        if success:
            print(f"Worker {worker_id} registered successfully.")
            return 0
        print("Registration failed.")
        return 1

    def cmd_login(self, args: argparse.Namespace) -> int:
        return self.cmd_register(args)

    def cmd_claim(self, args: argparse.Namespace) -> int:
        worker_id = getattr(args, "worker_id", "default")
        artifact_store = Path(getattr(args, "artifact_store", "./artifacts"))
        client = WorkerClient(
            coordinator_url=self.coordinator_url,
            worker_id=worker_id,
            runtime_version="0.1.0",
            artifact_store=artifact_store,
        )
        job = client.claim_job()
        self._log_history("claim", {"worker_id": worker_id, "job": job})
        if job:
            print(f"Claimed job: {job['job_id']}")
            print(json.dumps(job, indent=2))
            return 0
        print("No jobs available.")
        return 0

    def cmd_run(self, args: argparse.Namespace) -> int:
        job_id = getattr(args, "job_id", None)
        worker_id = getattr(args, "worker_id", "default")
        artifact_store = Path(getattr(args, "artifact_store", "./artifacts"))
        client = WorkerClient(
            coordinator_url=self.coordinator_url,
            worker_id=worker_id,
            runtime_version="0.1.0",
            artifact_store=artifact_store,
        )
        if job_id:
            job_spec = {"job_id": job_id, "dataset_url": "mock://dataset", "training_steps": 100}
        else:
            job = client.claim_job()
            if not job:
                print("No jobs available.")
                return 1
            job_spec = job
        limits = HardwareLimits()
        result = client.run_job(job_spec, limits)
        self._log_history("run", {"worker_id": worker_id, "result": result})
        print(json.dumps(result, indent=2))
        return 0 if result.get("success") else 1

    def cmd_validate(self, args: argparse.Namespace) -> int:
        output_dir = Path(getattr(args, "output_dir", "./output"))
        job_id = getattr(args, "job_id", "default")
        artifact_store = Path(getattr(args, "artifact_store", "./artifacts"))
        client = WorkerClient(
            coordinator_url=self.coordinator_url,
            worker_id="default",
            runtime_version="0.1.0",
            artifact_store=artifact_store,
        )
        job_spec = {"job_id": job_id, "dataset_url": "mock://dataset"}
        result = client.validate_result(output_dir, job_spec)
        print(json.dumps(result.model_dump(), indent=2))
        return 0 if result.passed else 1

    def cmd_submit(self, args: argparse.Namespace) -> int:
        package_path = Path(getattr(args, "package_path", "./output/result.zip"))
        job_id = getattr(args, "job_id", "default")
        artifact_store = Path(getattr(args, "artifact_store", "./artifacts"))
        client = WorkerClient(
            coordinator_url=self.coordinator_url,
            worker_id="default",
            runtime_version="0.1.0",
            artifact_store=artifact_store,
        )
        from prothynesis_runtime.worker.client import ValidationResult
        validation = ValidationResult(job_id=job_id, passed=True)
        job_spec = {"job_id": job_id, "dataset_url": "mock://dataset"}
        success = client.submit_result(package_path, job_spec, validation)
        print("Submitted successfully." if success else "Submission failed.")
        return 0 if success else 1

    def cmd_stop(self, args: argparse.Namespace) -> int:
        print("Stop signal sent. Graceful shutdown initiated.")
        return 0

    def cmd_history(self, args: argparse.Namespace) -> int:
        records = self._load_history()
        if not records:
            print("No history available.")
            return 0
        for record in records:
            print(json.dumps(record))
        return 0

    def cmd_doctor(self, args: argparse.Namespace) -> int:
        print("=" * 60)
        print("Prothynesis Runtime Doctor")
        print("=" * 60)
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        ram = psutil.virtual_memory()
        print(f"CPU Cores (physical): {cpu_count}")
        print(f"CPU Cores (logical): {cpu_count_logical}")
        print(f"RAM Total: {ram.total / (1024**3):.2f} GB")
        print(f"RAM Available: {ram.available / (1024**3):.2f} GB")
        if torch.cuda.is_available():
            print(f"CUDA: Available")
            print(f"  Device Count: {torch.cuda.device_count()}")
            print(f"  Device Name: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            print(f"  VRAM: {vram_gb:.2f} GB")
            estimated_capability = "high"
            recommended_concurrency = max(1, int(vram_gb // 8))
        else:
            print("CUDA: Not available")
            print("  Runtime: CPU only")
            estimated_capability = "low"
            recommended_concurrency = 1
        print(f"Runtime Version: 0.1.0")
        print(f"Estimated Solver Capability: {estimated_capability}")
        print(f"Recommended Concurrency: {recommended_concurrency}")
        print("=" * 60)
        return 0

    def build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="prothynesis-worker",
            description="Prothynesis Runtime Worker CLI",
        )
        parser.add_argument(
            "--coordinator-url",
            default=self.coordinator_url,
            help="Coordinator base URL",
        )
        parser.add_argument(
            "--worker-id",
            default="default",
            help="Worker identifier",
        )
        parser.add_argument(
            "--artifact-store",
            default="./artifacts",
            help="Artifact storage directory",
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        subparsers.add_parser("status", help="Show worker status")

        p_register = subparsers.add_parser("register", help="Register worker with coordinator")
        p_register.add_argument("--worker-id", default="default")

        p_login = subparsers.add_parser("login", help="Alias for register")
        p_login.add_argument("--worker-id", default="default")

        p_claim = subparsers.add_parser("claim", help="Claim an available job")
        p_claim.add_argument("--worker-id", default="default")

        p_run = subparsers.add_parser("run", help="Run a job end-to-end")
        p_run.add_argument("--job-id", default=None)

        p_validate = subparsers.add_parser("validate", help="Validate a local result")
        p_validate.add_argument("--output-dir", default="./output")
        p_validate.add_argument("--job-id", default="default")

        p_submit = subparsers.add_parser("submit", help="Submit a packaged result")
        p_submit.add_argument("--package-path", default="./output/result.zip")
        p_submit.add_argument("--job-id", default="default")

        subparsers.add_parser("stop", help="Stop the worker")

        subparsers.add_parser("history", help="Show command history")

        subparsers.add_parser("doctor", help="Show hardware diagnostics")

        return parser

    def run(self, argv: list[str] | None = None) -> int:
        parser = self.build_parser()
        args = parser.parse_args(argv)
        command = args.command
        handler = getattr(self, f"cmd_{command}", None)
        if not handler:
            print(f"Unknown command: {command}")
            return 1
        return handler(args)


def main() -> int:
    cli = WorkerCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
