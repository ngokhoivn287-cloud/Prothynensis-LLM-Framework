# Prothynesis Architecture

## Overview

Prothynesis is an experimental open-source AI system built around **MMNs 3.6** (Massive Model Networks) — a population of specialized Solvers that collectively solve complex reasoning tasks.

## Core Concepts

### Solver
An individual model (~150M parameters) with two capability layers:
- **Core General Intelligence** (~30M): instruction following, reasoning, planning, tool use, computer use, CRW protocol
- **Specialized Mastery** (~105M): deep expertise in a specific domain

### MMNs (Massive Model Networks)
The collective intelligence emerging from a population of specialized Solvers. The population provides breadth; each Solver provides depth.

### CRW (Collective Reasoning Web)
A structured communication protocol enabling Solvers to exchange hypotheses, evidence, arguments, verification results, plans, and tool observations.

## Architecture Layers

```
┌─────────────────────────────────────────────┐
│              ULTIMATE                       │  Final collective reasoning / synthesis
├─────────────────────────────────────────────┤
│               MASTER                        │  High-level strategy, conflict resolution
├─────────────────────────────────────────────┤
│               CHIEF                         │  Cross-specialty synthesis
├─────────────────────────────────────────────┤
│            ORCHESTRAL                       │  Coordinate multiple Solvers in same domain
├─────────────────────────────────────────────┤
│              SOLVER                         │  Deep expert reasoning in assigned domain
└─────────────────────────────────────────────┘
```

## Repository Structure

```
Prothynensis/
├── modular-moe/           # Core MoMMs/MoE research framework (historical)
├── prothynesis-runtime/   # Local inference + community worker
├── scripts/               # Maintained utility scripts
├── docs/                  # Project documentation
├── examples/              # Usage examples
└── tests/                 # Integration tests
```

## Development Status

**Current**: Prothynesis 3.1 / MMNs 3.6 (design phase)
- Core Solver architecture: In development
- Data allocation system: In development
- Runtime package: Usable for local inference
- Community worker: MVP functional
- Population training: Not yet implemented

See the Roadmap section above for planned milestones.