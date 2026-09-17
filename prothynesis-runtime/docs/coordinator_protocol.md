# Coordinator Protocol

## Overview

The Coordinator Protocol defines the structured message format for communication between training workers and the coordinator service. All messages are JSON-encoded and exchanged over HTTP. The protocol is designed to be deterministic, auditable, and resistant to arbitrary code execution.

## Registration

Workers register with the coordinator by sending their hardware report and runtime version.

### Request

```json
POST /workers/register
Content-Type: application/json

{
  "worker_id": "worker_001",
  "runtime_version": "0.1.0",
  "hardware": {
    "cpu_count": 4,
    "cpu_count_logical": 8,
    "ram_total_gb": 16.0,
    "ram_available_gb": 8.0,
    "cuda_available": true,
    "cuda_device_count": 1,
    "cuda_device_name": "NVIDIA RTX 4090",
    "vram_gb": 24.0
  }
}
```

### Response

```json
{
  "status": "registered",
  "worker_id": "worker_001",
  "assigned_at": "2024-01-01T00:00:00Z"
}
```

## Job Advertisement

The coordinator advertises available jobs to registered workers.

### Request

```json
GET /workers/{worker_id}/jobs
```

### Response

```json
{
  "jobs": [
    {
      "job_id": "job_2024_0000001",
      "solver_id": "1234567",
      "runtime_version": "0.1.0",
      "model_architecture": "cpu",
      "token_budget": 10000000,
      "dataset_url": "https://storage.example.com/datasets/ds_001.pt",
      "training_config": {
        "model_architecture": "cpu",
        "seed": 42,
        "token_budget": 10000000,
        "sequence_length": 1024
      },
      "validation_config": {
        "validation_steps": 6,
        "max_new_tokens": 256
      },
      "signature": "sig_abc123"
    }
  ]
}
```

## Job Claiming

Workers claim jobs by sending a claim request.

### Request

```json
POST /workers/{worker_id}/claim
```

### Response

```json
{
  "status": "claimed",
  "job_id": "job_2024_0000001",
  "worker_id": "worker_001",
  "claimed_at": "2024-01-01T00:00:00Z",
  "dataset_url": "https://storage.example.com/datasets/ds_001.pt"
}
```

## Result Submission

Workers submit trained solver results as packaged artifacts.

### Request

```json
POST /jobs/{job_id}/submit
Content-Type: multipart/form-data

{
  "package": "<binary zip file>",
  "worker_id": "worker_001",
  "validation_result": {
    "passed": true,
    "score": 0.95,
    "checks": [
      {"name": "checkpoint_exists", "passed": true},
      {"name": "token_budget", "passed": true}
    ]
  }
}
```

### Response

```json
{
  "status": "received",
  "job_id": "job_2024_0000001",
  "verified": true
}
```

## Verification

The coordinator verifies submitted packages before accepting results.

### Process

1. Compute SHA256 checksum of the received package
2. Compare against the checksum in package metadata
3. Verify manifest structure and required fields
4. Run validation checks from the worker's submission
5. Accept or reject based on verification results

### Response

```json
{
  "status": "ACCEPTED",
  "job_id": "job_2024_0000001",
  "verified_at": "2024-01-01T00:00:00Z"
}
```

Or on failure:

```json
{
  "status": "REJECTED",
  "job_id": "job_2024_0000001",
  "reason": "checksum mismatch"
}
```

## Protocol Messages

All protocol messages follow these conventions:
- Content-Type is `application/json` for JSON payloads
- Timestamps use ISO 8601 format with UTC timezone
- Errors include a `reason` field with a human-readable description
- Successful responses include a `status` field with a machine-readable status code

## Error Handling

Errors are returned with appropriate HTTP status codes:

| HTTP Status | Meaning |
|-------------|---------|
| 200 | Success |
| 400 | Invalid request format or missing fields |
| 401 | Worker not registered or invalid credentials |
| 403 | Job not available for this worker |
| 404 | Job or worker not found |
| 409 | Job already claimed or completed |
| 500 | Internal coordinator error |

Error response body:

```json
{
  "status": "error",
  "reason": "Job not available: already claimed by worker_002",
  "job_id": "job_2024_0000001"
}
```

## Structured Job Specs

Jobs are defined using structured job specifications, not arbitrary scripts. The job spec includes:

- `job_id`: Unique identifier matching pattern `job_YYYY_NNNNNNN`
- `solver_id`: Target solver identifier matching pattern `\d{7}`
- `runtime_version`: Semantic version of the runtime required
- `model_architecture`: Target hardware architecture (cpu, cuda, etc.)
- `dataset_url`: URL to download the training dataset
- `token_budget`: Maximum tokens for training
- `training_config`: Structured configuration for the training run
- `validation_config`: Structured configuration for validation
- `signature`: Cryptographic signature of the job spec

This structure ensures that:
- All jobs are deterministic and reproducible
- Jobs can be validated without executing arbitrary code
- Jobs can be audited and inspected before training

## No Arbitrary Code Execution

The protocol explicitly forbids arbitrary code execution:
- Jobs contain no executable code, only structured configuration
- Training is performed using predefined, deterministic operations
- Workers cannot execute arbitrary commands from job specifications
- All file operations are constrained to designated artifact directories

This design ensures that workers can safely execute any valid job without risk of code injection or malicious behavior.
