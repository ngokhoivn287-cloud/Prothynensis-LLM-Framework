# Prothynesis Runtime

## Overview

Prothynesis Runtime is a local-first system for model inference and community solver training. It is designed to run entirely on your machine with optional connectivity to coordinate training contributions.

The runtime provides:
- Local model inference without external API calls
- Community training worker for contributing solver checkpoints
- Hardware-aware resource management
- Secure job validation and artifact integrity checks

## Architecture

Prothynesis 3.6 uses **MMNs 7.x** (Massive Model Networks) — a population of ~150M parameter Solvers. Each Solver has two capability layers:

1. **Core General Intelligence** (shared foundation): instruction following, reasoning, planning, tool use, computer use, CRW protocol
2. **Specialized Mastery** (deep domain expertise): each Solver masters a specific domain like systems programming, mathematics, security, etc.

The population provides **BREADTH**; each Solver provides **DEPTH**.

See [MMNs 7.x Architecture](mmns_7_architecture.md) for the full design.

## Installation

Install from source:

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

GPU support requires the optional GPU dependencies:

```bash
pip install -e ".[gpu]"
```

## Local Inference

The runtime operates in **local-only mode** by default. Models are loaded and executed on your machine. No data is sent to remote servers unless you explicitly enable community training features.

To run inference:

```bash
prothynesis run --model <path_to_checkpoint> --prompt "Hello world"
```

Or start an interactive chat session:

```bash
prothynesis chat --model <path_to_checkpoint>
```

## Model Management

Models are stored locally in the runtime storage directory. The runtime supports:
- Downloading models from Hugging Face Hub
- Managing model metadata and shard information
- Format conversion between safetensors and PyTorch formats

## Hardware Detection

The runtime automatically detects your hardware configuration:
- CPU count and available cores
- RAM total and available memory
- GPU presence, model, and VRAM
- CUDA availability and device count

Hardware information is reported when registering as a training worker and used to match jobs to suitable machines.

## Settings

Configuration is managed through YAML files and environment variables. Key settings include:
- `runtime_version`: The version of the runtime
- `coordinator_url`: URL of the optional coordinator service
- `hardware_limits`: Conservative defaults for GPU utilization, VRAM, and temperature
- `pause_while_user_active`: Pause training when the system is in use

## CLI Reference

The `prothynesis` command provides the following subcommands:

| Command | Description |
|---------|-------------|
| `prothynesis run --model <path> --prompt "<text>"` | Run single inference request |
| `prothynesis chat --model <path>` | Interactive chat session |
| `prothynesis api` | Start local OpenAI-compatible API server |
| `prothynesis web` | Start local web UI |
| `prothynesis info` | Show runtime status |
| `prothynesis hardware` | Show hardware information |
| `prothynesis model list` | List loaded models |
| `prothynesis model import <path>` | Import a model checkpoint |
| `prothynesis model verify <path>` | Verify a model checkpoint |
| `prothynesis model discover [path]` | Discover models in a directory |
| `prothynesis worker status` | Show worker status |
| `prothynesis doctor` | Run diagnostics |

## API Reference

The runtime exposes a local API server for programmatic access:

```bash
prothynesis api --host 127.0.0.1 --port 8000
```

Key endpoints:
- `POST /v1/chat/completions` - Run chat completion with a loaded model
- `GET /v1/models` - List available models
- `GET /v1/status` - Get runtime status
- `GET /health` - Health check

The API is OpenAI-compatible for chat completions. Load a model first via `prothynesis model import <path>` or by passing `model_path` to the orchestrator.

## Web UI

The optional Web UI provides a browser-based interface for runtime management.

Start the Web UI:

```bash
prothynesis web --host 127.0.0.1 --port 3000
```

The Web UI binds to `127.0.0.1` by default and is accessible only from the local machine.

## Storage Locations

Runtime data is stored in platform-specific locations:

| Data Type | Location |
|-----------|----------|
| Models | `<storage>/models/` |
| Training artifacts | `<storage>/artifacts/` |
| Worker cache | `<storage>/cache/` |
| Logs | `<storage>/logs/` |

The base storage directory can be configured via the `PROTHYNESIS_STORAGE` environment variable.

## Troubleshooting

### CUDA Out of Memory
Reduce `max_vram_usage_gb` in hardware limits or use CPU-only mode.

### Model Download Fails
Check your internet connection and Hugging Face Hub access. Models can be pre-downloaded manually.

### Worker Registration Fails
Ensure the coordinator URL is reachable and your runtime version is supported.

### Authentication Issues
Run `prothynesis auth status` to verify your GitHub connection. Re-authenticate with `prothynesis auth login`.
