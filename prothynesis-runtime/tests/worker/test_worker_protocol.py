import pytest

from prothynesis_runtime.worker.protocol import WorkerProtocol, WorkerState


def test_worker_protocol_initial_state():
    protocol = WorkerProtocol("worker_001", "0.1.0")
    assert protocol.get_state() == WorkerState.IDLE


def test_worker_protocol_register_payload():
    protocol = WorkerProtocol("worker_001", "0.1.0")
    hardware_report = {
        "cpu_count": 4,
        "ram_total_gb": 16.0,
        "vram_gb": 8.0,
        "cuda_available": True,
    }
    payload = protocol.register(hardware_report)
    assert payload["worker_id"] == "worker_001"
    assert payload["runtime_version"] == "0.1.0"
    assert payload["hardware"] == hardware_report
    assert "timestamp" in payload
    assert protocol.get_state() == WorkerState.REGISTERED


def test_worker_protocol_heartbeat_payload():
    protocol = WorkerProtocol("worker_001", "0.1.0")
    hardware_report = {
        "cpu_count": 4,
        "ram_gb": 16.0,
        "vram_gb": 8.0,
        "temperature_c": 65.0,
    }
    protocol.register(hardware_report)
    payload = protocol.heartbeat()
    assert payload["worker_id"] == "worker_001"
    assert payload["state"] == "registered"
    assert "timestamp" in payload
    assert payload["vram"] == 8.0
    assert payload["ram"] == 16.0


def test_worker_protocol_claim_job_payload():
    protocol = WorkerProtocol("worker_001", "0.1.0")
    job_spec = {
        "job_id": "job_2024_0000001",
        "solver_id": "1234567",
        "dataset_url": "http://example.com/data",
    }
    payload = protocol.claim_job(job_spec)
    assert payload["worker_id"] == "worker_001"
    assert payload["job_id"] == "job_2024_0000001"
    assert payload["status"] == "claimed"
    assert "timestamp" in payload
    assert protocol.get_state() == WorkerState.CLAIMED


def test_worker_protocol_submit_result_payload():
    protocol = WorkerProtocol("worker_001", "0.1.0")
    validation_result = {"passed": True, "score": 0.95}
    payload = protocol.submit_result("/path/to/package.zip", validation_result)
    assert payload["worker_id"] == "worker_001"
    assert payload["package_path"] == "/path/to/package.zip"
    assert payload["validation_result"] == validation_result
    assert "timestamp" in payload
    assert protocol.get_state() == WorkerState.IDLE


def test_worker_protocol_state_transitions():
    protocol = WorkerProtocol("worker_001", "0.1.0")
    assert protocol.get_state() == WorkerState.IDLE
    protocol.set_state(WorkerState.REGISTERED)
    assert protocol.get_state() == WorkerState.REGISTERED
    protocol.set_state(WorkerState.RUNNING)
    assert protocol.get_state() == WorkerState.RUNNING
    protocol.set_state(WorkerState.IDLE)
    assert protocol.get_state() == WorkerState.IDLE
