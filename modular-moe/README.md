# Modular MoE / MoMMs Research Framework

**Part of Prothynesis 3.1 — MMNs 3.6 Architecture**

This is the core research framework for modular pretraining of Mixture-of-Experts (MoE) language models, evolving toward the Mix of Many Models (MoMMs) architecture used in Prothynesis MMNs.

## Overview

The framework implements a **modular pretraining pipeline** for MoE models:

1. **Stage A**: Dataset preparation → cleaning → deduplication → tokenization → sharding
2. **Stage B**: Independent expert pretraining (each expert on its own data shard)
3. **Stage C**: Expert validation (independent evaluation)
4. **Stage D**: MoE assembly (load pretrained experts into shared architecture)
5. **Stage E**: Router training (freeze experts, train router)
6. **Stage F**: Joint training (unfreeze all, continue pretraining)
7. **Stage G**: Comprehensive evaluation

## Key Research Idea

> **An expert is an independently trainable, independently checkpointable module that can later participate in a shared MoE model.**

This abstraction enables scaling from 1.6B → 6.4B → 25B → 100B+ → 256B total parameters using the same framework.

## Architecture (Legacy MoE — See MMNs 3.6 for Current)

The current codebase implements a **conventional MoE architecture**:
- Decoder-only Transformer with MoE layers replacing standard FFN blocks
- Configurable experts: 32, 128, 512, or thousands
- Top-K routing (default: Top-2) with learned router
- SwiGLU expert FFNs targeting ~50M parameters each
- Shared components: Attention, embeddings, LM head, layer norms
- Expert components: Independent FFN weights per expert
- Router components: Learned routing network

**Note**: This MoE architecture is the historical foundation. The current Prothynesis 3.1 / MMNs 3.6 architecture uses independent Solver models (~150M each) with a shared Core General Intelligence layer and specialized mastery layers, coordinated through CRW (Collective Reasoning Web). See `docs/architecture.md` for the current architecture.

## Parameter Breakdown (1.6B Prototype)

```
Embedding:          134,217,728  (vocab=50304, dim=2664)
Attention:          536,870,912  (32 layers, 2664 dim, 32 heads)
Shared:             268,435,456  (layer norms, etc.)
Each Expert:         50,331,648  (SwiGLU, hidden=10752)
All Experts (32):  1,610,612,736
Router:              33,554,432  (2664 → 32)
LM Head:            134,217,728  (tied or untied)
Total:            2,717,908,992  (~2.7B with untied head)
                     2,583,691,264  (~2.6B with tied head)
```

## Installation

```bash
pip install -e ".[dev]"
```

## Configuration

Configuration via YAML files in `configs/`:
- `custom.yaml` — Example configuration
- `datasets.yaml` — Dataset registry

Additional configurations for historical experiments are in the research archive.

## Project Structure

```
modular-moe/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── custom.yaml
│   └── datasets.yaml
├── src/
│   ├── cli.py
│   ├── data/
│   ├── evaluation/
│   ├── gui/
│   ├── model/
│   ├── momm/           # MoMMs architecture components
│   ├── pmo/
│   ├── runtime/
│   ├── training/
│   └── utils/
├── tests/
└── checkpoints/        # Ignored by Git (generated)
```

## Development Status

**Experimental Research Framework** — Not production ready.

- Core MoE training pipeline: Functional
- MoMMs components (momm/): In development
- Distributed training: Not implemented
- Expert parallelism: Not implemented
- FlashAttention: Not integrated

See the [architecture documentation](../docs/architecture.md) for the MMNs 3.6 design.

## License

MIT License — See LICENSE file for details.