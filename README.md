# Prothynesis

**Prothynesis 3.1** — **MMNs 3.6** (Massive Model Networks)

An experimental open-source AI system built around a population of specialized Solver models that collectively solve complex reasoning tasks.

## Overview

Prothynesis is a research project exploring **Massive Model Networks (MMNs)** — an architecture where collective intelligence emerges from a network of highly capable specialists rather than a single large generalist model.

### Core Concepts

- **Solver**: An individual model (~150M parameters) with two capability layers:
  - **Core General Intelligence** (~30M): instruction following, reasoning, planning, tool use, computer use, CRW protocol
  - **Specialized Mastery** (~105M): deep expertise in a specific domain (systems, math, security, software engineering, etc.)

- **MMNs (Massive Model Networks)**: The collective intelligence emerging from a population of specialized Solvers. The population provides breadth; each Solver provides depth.

- **CRW (Collective Reasoning Web)**: A structured communication protocol enabling Solvers to exchange hypotheses, evidence, arguments, verification results, plans, and tool observations.

### Architecture Layers

```
ULTIMATE       → Final collective reasoning / synthesis
MASTER         → High-level strategy, conflict resolution
CHIEF          → Cross-specialty synthesis
ORCHESTRAL     → Coordinate multiple Solvers in same domain
SOLVER         → Deep expert reasoning in assigned domain
```

## Repository Structure

```
Prothynensis/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
├── SECURITY.md                  # Security policy
├── CODE_OF_CONDUCT.md           # Code of conduct
├── .gitignore
├── docs/
│   └── architecture.md          # Architecture documentation
├── modular-moe/                 # Core MoE/MoMMs research framework
│   ├── README.md
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── configs/
│   ├── src/
│   └── tests/
├── prothynesis-runtime/         # Local inference + community worker package
│   ├── README.md
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── requirements-gpu.txt
│   ├── src/
│   ├── tests/
│   ├── docs/
│   ├── examples/
│   └── packaging/
└── scripts/                     # Maintained utility scripts
```

## Getting Started

### Runtime (Local Inference)

```bash
cd prothynesis-runtime
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
prothynesis chat
```

### Development (Core Framework)

```bash
cd modular-moe
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Development Status

**Current: Prothynesis 3.1 / MMNs 3.6 (Design & Early Implementation)**

| Component | Status |
|-----------|--------|
| Core Solver architecture (150M) | In development |
| Data allocation system | In development |
| Runtime package | Usable for local inference |
| Community worker | MVP functional |
| Population training | Not yet implemented |
| CRW protocol | In development |
| Hierarchy (Orchestral/Chief/Master/Ultimate) | Planned |

## Documentation

- [Architecture](docs/architecture.md) — MMNs 3.6 design
- [Runtime](prothynesis-runtime/docs/runtime.md) — Local inference guide
- [Worker](prothynesis-runtime/docs/worker.md) — Community training
- [Modular MoE Framework](modular-moe/README.md) — Research framework

## Dataset & Checkpoints

Training data and experimental checkpoints are stored **externally** (not in Git).

Configure the dataset root via the `PROTHYNESIS_DATA_ROOT` environment variable:

**Windows (PowerShell):**
```powershell
$env:PROTHYNESIS_DATA_ROOT = "D:\path\to\datasets"
```

**Windows (cmd):**
```cmd
set PROTHYNESIS_DATA_ROOT=D:\path\to\datasets
```

**Linux / WSL / macOS (bash/zsh):**
```bash
export PROTHYNESIS_DATA_ROOT=/path/to/datasets
```

Defaults (when unset):
- Windows: `D:\NgoPROJECT\ProthynensisDatasets`
- WSL/Linux: `/mnt/d/NgoPROJECT/ProthynensisDatasets`

The external directory contains:
- Raw and processed datasets
- Experiment artifacts (population_scaling, crw_real, oracle_gap, etc.)
- Training checkpoints

This directory is **not** part of the Git repository.

## Testing

```bash
# Core framework tests
cd modular-moe
pytest tests/ -v

# Runtime tests
cd prothynesis-runtime
pytest tests/ -v
```

## License

MIT License — See LICENSE file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Security

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
