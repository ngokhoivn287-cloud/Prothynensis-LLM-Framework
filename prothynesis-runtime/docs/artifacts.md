# Artifact System

## Overview

The Artifact System manages the packaging, storage, verification, and distribution of solver training results. Artifacts include trained model checkpoints, training metadata, validation results, and package metadata. The system ensures integrity through cryptographic checksums and supports multiple storage backends.

## Solver Result Package

A solver result package contains all outputs from a training run:

```
solver_<solver_id>/
  checkpoint/
    <checkpoint_file>.pt
  manifest.json
  validation.json
  checksum.sha256
```

### Package Contents

| File | Description |
|------|-------------|
| `checkpoint/<file>.pt` | Trained model weights |
| `manifest.json` | Training metadata including solver_id, job_id, and checkpoint hash |
| `validation.json` | Validation results and checks |
| `checksum.sha256` | SHA256 checksum of the checkpoint file |

## Packaging Format

Packages are created using the `PMOPackage` or `package_solver_result` utilities. The packaging process:

1. Copies the checkpoint file to a structured output directory
2. Computes SHA256 checksum of the checkpoint
3. Writes `manifest.json` with training metadata and checkpoint hash
4. Writes `validation.json` with validation results
5. Writes `checksum.sha256` with the checkpoint hash
6. Optionally compresses the directory into a zip archive

### Example manifest.json

```json
{
  "solver_id": "1234567",
  "job_id": "job_2024_0000001",
  "runtime_version": "0.1.0",
  "training_metadata": {
    "final_loss": 0.05,
    "tokens_seen": 1000000,
    "loss_history": [0.9, 0.8, 0.5, 0.05]
  },
  "contributor_metadata": {
    "username": "testuser",
    "github_id": 42
  },
  "checkpoint_hash": "abc123def456..."
}
```

### Example validation.json

```json
{
  "validation_status": "passed",
  "checks": [
    {"name": "checkpoint_exists", "passed": true, "message": "Checkpoint found"},
    {"name": "token_budget", "passed": true, "message": "Within tolerance"},
    {"name": "checksum", "passed": true, "message": "Checksum matches"}
  ],
  "timestamp": "2024-01-01T00:00:00"
}
```

## Checksums

All artifacts are verified using SHA256 checksums:

- **Checkpoint checksum**: Computed during packaging and stored in `checksum.sha256`
- **Package checksum**: Computed over the entire package zip for submission verification
- **Verification**: Checksums are verified on download, extraction, and submission

Checksum verification prevents:
- Corrupted downloads
- Tampered artifacts
- Incomplete transfers
- Accidental file modifications

## Storage Backends

The artifact system supports multiple storage backends:

| Backend | Description |
|---------|-------------|
| Local filesystem | Default, stores artifacts in the local storage directory |
| HTTP(S) remote | Downloads artifacts from HTTP(S) URLs |
| Custom store | Pluggable storage backend via the `ArtifactStore` interface |

Storage locations are configured in the runtime settings and can be changed per-artifact-type.

## Upload/Download

Artifacts are transferred using standard HTTP mechanisms:

### Download

```python
from prothynesis_runtime.artifacts.store import ArtifactStore

store = ArtifactStore()
path = store.download("https://storage.example.com/artifacts/checkpoint.pt")
```

### Upload

```python
from prothynesis_runtime.artifacts.packager import compute_package_checksum

checksum = compute_package_checksum("solver_1234567/package.zip")
# Upload via HTTP PUT or multipart form
```

## Resumable Upload

For large artifacts, the system supports resumable uploads:

1. The client requests an upload URL from the storage backend
2. The client uploads the artifact in chunks
3. If the upload is interrupted, the client can resume from the last confirmed chunk
4. The server verifies the complete upload using the expected checksum

This is particularly useful for large model checkpoints that may take significant time to transfer.

Resumable uploads are used when:
- The artifact size exceeds 100MB
- The network connection is unreliable
- The storage backend supports chunked uploads
