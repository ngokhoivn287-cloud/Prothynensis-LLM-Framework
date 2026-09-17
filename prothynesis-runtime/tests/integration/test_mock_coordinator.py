import os
import tempfile

import pytest

from prothynesis_runtime.coordinator.mock import MockCoordinator


def test_mock_coordinator_register_worker():
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = MockCoordinator(tmpdir)
        worker_payload = {
            "worker_id": "worker_001",
            "runtime_version": "0.1.0",
            "hardware": {"cpu_count": 4, "ram_total_gb": 16.0},
        }
        result = coordinator.register_worker(worker_payload)
        assert result["worker_id"] == "worker_001"
        assert result["status"] == "active"
        assert "registered_at" in result


def test_mock_coordinator_advertise_jobs_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = MockCoordinator(tmpdir)
        jobs = coordinator.advertise_jobs()
        assert jobs == []


def test_mock_coordinator_claim_job_transitions():
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = MockCoordinator(tmpdir)
        coordinator._jobs["job_001"] = {"job_id": "job_001", "status": "pending"}
        result = coordinator.claim_job("job_001", "worker_001")
        assert result["status"] == "claimed"
        assert result["job_id"] == "job_001"
        assert result["worker_id"] == "worker_001"
        assert coordinator._jobs["job_001"]["status"] == "claimed"


def test_mock_coordinator_receive_result():
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = MockCoordinator(tmpdir)
        coordinator._jobs["job_001"] = {"job_id": "job_001", "status": "pending"}
        package_metadata = {"checksum": "abc123"}
        validation_result = {"passed": True, "score": 0.95}
        result = coordinator.receive_result(
            "job_001", "worker_001", package_metadata, validation_result
        )
        assert result["status"] == "received"
        assert "job_001" in coordinator._results
        assert coordinator._jobs["job_001"]["status"] == "completed"


def test_mock_coordinator_verify_result_accepted():
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = MockCoordinator(tmpdir)
        coordinator._jobs["job_001"] = {"job_id": "job_001", "status": "completed"}
        package_path = os.path.join(tmpdir, "package.zip")
        package_data = b"data"
        with open(package_path, "wb") as f:
            f.write(package_data)
        import hashlib
        sha = hashlib.sha256()
        sha.update(package_data)
        coordinator._results["job_001"] = {
            "package_metadata": {"checksum": sha.hexdigest()}
        }
        result = coordinator.verify_result("job_001", package_path)
        assert result["status"] == "ACCEPTED"


def test_mock_coordinator_get_population():
    with tempfile.TemporaryDirectory() as tmpdir:
        coordinator = MockCoordinator(tmpdir)
        coordinator._population = [
            {"solver_id": "solver_001", "status": "accepted"},
            {"solver_id": "solver_002", "status": "accepted"},
        ]
        population = coordinator.get_population()
        assert len(population) == 2
        assert population[0]["solver_id"] == "solver_001"
