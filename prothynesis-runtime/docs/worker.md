# Community Training Worker

## Overview

The Community Training Worker is an optional component of the Prothynesis Runtime. It allows you to contribute computing resources to train Solver models for the MMNs 7.x population. Participation is entirely voluntary and can be stopped at any time.

The worker runs locally on your machine, downloads training jobs, executes training according to the job specification, validates the result, and submits the trained checkpoint back to the coordinator.

### MMNs 7.x Training Context

In MMNs 7.x, each Solver is ~150M parameters with two capability layers:

1. **Core General Intelligence** — shared foundation (instruction following, reasoning, planning, tool use, computer use, CRW)
2. **Specialized Mastery** — deep domain expertise (systems, math, security, software engineering, etc.)

Training jobs specify the Solver's specialization profile, dataset shard, and token budget. The worker trains the Solver to deeply master its assigned domain while preserving general capability.

## Prerequisites

- Prothynesis Runtime installed and configured
- Python 3.11 or later
- Optional: CUDA-capable GPU with sufficient VRAM (12GB+ recommended for 150M Solver training)
- Optional: GitHub account for attribution

## Registration

To register as a worker:

```bash
prothynesis worker register
```

The registration process:
1. Reports your hardware configuration to the coordinator
2. Receives a worker ID
3. Stores local worker state

The worker will only accept jobs that match its reported hardware capabilities.

## Hardware Requirements

The worker enforces conservative hardware limits to avoid disrupting your system:

- **GPU Utilization**: Capped at 90% by default
- **VRAM Usage**: Configurable limit (e.g., 12GB max for 150M Solver)
- **Temperature**: Training pauses if GPU temperature exceeds the configured threshold
- **Concurrent Solvers**: Limited to 1 by default
- **User Activity**: Training can pause when you are actively using the system

These limits are enforced automatically during training.

## Worker Commands

| Command | Description |
|---------|-------------|
| `prothynesis worker register` | Register with the coordinator |
| `prothynesis worker start` | Start the worker loop |
| `prothynesis worker stop` | Stop the worker gracefully |
| `prothynesis worker status` | Show current worker state |
| `prothynesis worker pause` | Pause after current job |
| `prothynesis worker resume` | Resume from pause |

## Worker States

The worker transitions through the following states:

| State | Description |
|-------|-------------|
| `idle` | Not registered or waiting for work |
| `registered` | Registered with coordinator, waiting for jobs |
| `claimed` | Job claimed, preparing to train |
| `running` | Actively training the solver |
| `validating` | Running post-training validation |
| `submitting` | Packaging and submitting results |
| `paused` | Paused due to user activity or hardware limits |
| `error` | Encountered an error, will retry |

## Hardware Limits

Configure hardware limits in your runtime settings:

```yaml
hardware_limits:
  max_gpu_utilization: 0.9
  max_vram_usage_gb: 8.0
  max_temperature_c: 85
  max_concurrent_solvers: 1
  pause_while_user_active: true
```

The worker will automatically:
- Reduce batch size if VRAM limits are approached
- Pause training if temperature thresholds are exceeded
- Skip training if user activity is detected (when enabled)

## Training Flow

1. **Heartbeat**: The worker periodically sends heartbeats to the coordinator with hardware status
2. **Job Advertisement**: The coordinator sends available jobs matching the worker's capabilities
3. **Job Claiming**: The worker claims a job and downloads the dataset
4. **Training**: The worker trains the solver using the job specification
5. **Validation**: The worker validates the checkpoint against the job requirements
6. **Packaging**: The worker packages the checkpoint, metadata, and validation results
7. **Submission**: The worker submits the package to the coordinator

## Validation

Before submission, the worker runs validation checks:
- Checkpoint file integrity
- Architecture compatibility
- Token budget within tolerance
- Training metadata completeness
- Checksum verification

Only jobs that pass all validation checks are submitted.

## Submission

The packaged result includes:
- `model.pt` - The trained checkpoint
- `metadata.json` - Training metrics and configuration
- `validation.json` - Validation results
- `package_metadata.json` - Package information including checksums

The coordinator verifies the package checksum before accepting the result.

## Attribution

When you register as a worker, you can optionally link your GitHub account. This allows:
- Your contributions to be tracked in the community leaderboard
- Attribution for solver checkpoints you train
- Recognition in model metadata

Attribution is completely optional and does not affect your ability to participate.
