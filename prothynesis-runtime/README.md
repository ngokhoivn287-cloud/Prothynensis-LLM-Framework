# Prothynesis Runtime

Open-source, self-contained runtime for local inference and community volunteer Solver training.

## Features

- **Local Inference**: Load `.pmo` models or direct checkpoints (`.pt`, `.safetensors`), run real text generation, OpenAI-compatible API
- **Community Training**: Optional worker mode for contributing compute to train Solvers (MVP)
- **Hardware Safety**: Conservative defaults, CPU/CUDA auto-detection
- **GitHub Integration**: Optional OAuth/App authorization for attribution (MVP)
- **Local-Only Mode**: Zero network required for inference

## Architecture

Prothynesis 3.1 uses **MMNs 3.6** (Massive Model Networks) — a population of ~150M parameter Solvers, each deeply specialized in a domain, coordinated through CRW (Collective Reasoning Web).

- **BREADTH** = population coverage
- **DEPTH** = individual Solver mastery
- **COLLECTIVE INTELLIGENCE** = MMNs

See [MMNs 3.6 Architecture](../docs/architecture.md) for details.

## Quick Start

### Install

```bash
git clone https://github.com/Prothynesis/prothynesis-runtime.git
cd prothynesis-runtime
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Local Inference

```bash
prothynesis chat
prothynesis api
```

### Community Training (Optional)

```bash
prothynesis worker register
prothynesis worker claim
prothynesis worker run
prothynesis worker validate
prothynesis worker submit
```

## Documentation

- [Runtime](docs/runtime.md)
- [Worker](docs/worker.md)
- [GitHub Integration](docs/github.md)
- [Security](docs/security.md)
- [Coordinator Protocol](docs/coordinator_protocol.md)
- [Artifacts](docs/artifacts.md)

## License

MIT