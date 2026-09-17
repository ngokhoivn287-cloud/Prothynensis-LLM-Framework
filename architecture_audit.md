# Architecture Audit Report: Prothynensis Modular MoE → MoMMs

**Date:** 2026-09-10  
**Auditor:** Lead ML Systems Engineer  
**Repository:** D:\NgoPROJECT\Prothynensis\modular-moe\

---

## Executive Summary

The current codebase implements a **conventional Mixture-of-Experts (MoE) architecture**, **not** a genuine Mix of Many Models (MoMMs) architecture. The codebase is well-structured and functional for MoE, but fundamentally incompatible with the MoMMs vision described in the project requirements.

---

## Current Architecture Analysis

### Overall Architecture Type
**Conventional MoE** — Single Transformer backbone with MoE layers replacing FFN blocks.

### Component Breakdown

| Component | Current Implementation | MoMMs Compatible? |
|-----------|------------------------|-------------------|
| **Model** | Single `MoELanguageModel` with shared Transformer backbone | ❌ No |
| **Experts** | `ExpertFFN` modules (SwiGLU FFNs) inside `MoELayer` | ❌ No |
| **Routing** | Top-K learned router per MoE layer | ⚠️ Partial |
| **Backbone** | Single shared Transformer (attention, embeddings, norms) | ❌ No |
| **LM Head** | Single shared LM head (tied or untied) | ❌ No |
| **Embeddings** | Single shared token + position embeddings | ❌ No |
| **Tokenizer** | Single shared tokenizer | ❌ No |
| **LM Head** | Single shared (tied or untied) | ❌ No |

### Training Pipeline (7 Stages)
| Stage | Description | MoMMs Compatible? |
|-------|-------------|-------------------|
| A | Dataset download + sharding | ✅ Compatible |
| B | Expert pretraining (per-shard) | ⚠️ Partial - trains FFN in mini-Transformer |
| C | Expert validation | ⚠️ Partial |
| D | MoE assembly (load expert weights) | ❌ No - loads into shared MoE |
| E | Router training (freeze experts) | ❌ No - trains router in MoE |
| F | Joint training (progressive unfreeze) | ❌ No - unfreezes shared MoE |
| G | Evaluation | ⚠️ Partial |

### Key Architectural Assumptions (MoE-centric)

1. **Single Transformer Backbone** — All "experts" share attention, embeddings, norms, LM head
2. **Experts as FFN Modules** — `ExpertFFN` is just a SwiGLU FFN, not an independent model
3. **Shared Experts Across Layers** — Same expert weights reused across all MoE layers
4. **MoE Layer = FFN Replacement** — MoE layer replaces FFN in Transformer block
4. **Top-K Routing Within Single Transformer** — Router selects expert FFNs within single Transformer
5. **Shared Embeddings/Head** — Single token/position embeddings, single LM head
6. **Single Tokenizer** — One tokenizer for all "experts"

---

## MoE vs MoMMs Gap Analysis

| Aspect | Current (MoE) | Required (MoMMs) | Gap |
|--------|---------------|------------------|-----|
| **Model Independence** | ❌ Shared backbone | Each model independently trainable | **Major** |
| **Model Architecture** | FFN experts in shared Transformer | Complete independent models (embeddings, attention, FFN, head) | **Major** |
| **Model Checkpointing** | Expert weights only | Full model checkpoints (embeddings, attention, FFN, head) | **Major** |
| **Routing** | Per-layer Top-K within single Transformer | Cross-model coordination | **Major** |
| **Fusion** | MoE dispatch/combine | Cross-model fusion | **Major** |
| **Training** | Expert pretraining in mini-Transformer | Independent model pretraining | **Major** |
| **Coordination** | Router in MoE layer | Model coordinator | **Major** |
| **Fusion** | MoE dispatch/combine | Cross-model fusion layer | **Major** |
| **Model Registry** | None | Dynamic model discovery | **Major** |
| **Tokenizer** | Single shared | Per-model or shared | **Medium** |
| **Checkpointing** | Expert weights only | Full model + coordinator + fusion | **Major** |

---

## Files Requiring Major Changes

### Core Model Architecture (HIGH PRIORITY)
| File | Current Role | Required Change |
|------|--------------|-----------------|
| `src/model/model.py` | MoE Language Model | Replace with MoMMs orchestrator |
| `src/model/moe.py` | MoE Layer with ExpertFFN | Replace with Model Coordinator + Fusion |
| `src/model/expert.py` | ExpertFFN (SwiGLU FFN) | Replace with independent Model class |
| `src/model/transformer.py` | Shared Transformer stack | Replace with Model Coordinator |
| `src/model/router.py` | MoE Router | Replace with Model Coordinator + Router |
| `src/model/attention.py` | Shared attention | Keep (used per-model) |

### Training Pipeline (HIGH PRIORITY)
| File | Current Role | Required Change |
|------|--------------|-----------------|
| `src/training/expert_trainer.py` | Expert pretraining in mini-Transformer | Independent model training |
| `src/training/router_trainer.py` | Router training in MoE | Coordinator training |
| `src/training/joint_trainer.py` | Joint MoE unfreeze | MoMMs integration training |
| `src/training/base_trainer.py` | Base trainer | Keep (adapt for MoMMs) |
| `src/training/checkpoint.py` | Expert/MoE checkpoints | MoMMs checkpoint format |

### Data Pipeline (MEDIUM PRIORITY)
| File | Current Role | Required Change |
|------|--------------|-----------------|
| `src/data/sharding.py` | Expert-centric sharding | Model-aware sharding |
| `src/data/dataset.py` | Dataset loading | Keep (adapt) |
| `src/data/tokenizer.py` | Single tokenizer | Multi-tokenizer support |

### Configuration (MEDIUM PRIORITY)
| File | Current Role | Required Change |
|------|--------------|-----------------|
| `configs/prototype_1p6b.yaml` | MoE config | MoMMs config |
| `configs/tiny_test.yaml` | Tiny MoE test | Tiny MoMMs test |
| `configs/datasets.yaml` | Dataset registry | Keep + model assignment |

### Scripts (MEDIUM PRIORITY)
| Script | Current Role | Required Change |
|--------|--------------|-----------------|
| `scripts/train_expert.py` | Train expert | Train independent model |
| `scripts/train_router.py` | Train MoE router | Train coordinator |
| `scripts/train_joint.py` | Joint MoE training | MoMMs integration |
| `scripts/launch_experts.py` | Launch expert training | Launch model training |
| `scripts/autotraining.py` | 7-stage pipeline | MoMMs pipeline |
| `scripts/evaluate.py` | MoE eval | MoMMs eval |

### New Files Required (HIGH PRIORITY)
| File | Purpose |
|------|---------|
| `src/momm/registry.py` | Model registry |
| `src/momm/coordinator.py` | Model coordinator |
| `src/momm/fusion.py` | Cross-model fusion |
| `src/momm/inference.py` | Native inference engine |
| `src/momm/__init__.py` | Package init |
| `gui.py` | ChatGPT-like GUI |
| `scripts/export_gguf.py` | GGUF export with validation |

---

## MoE-Specific Terminology to Replace

| Current Term | MoMMs Replacement |
|--------------|-------------------|
| `expert` / `expert_id` | `model` / `model_id` |
| `expert_ffn` / `ExpertFFN` | `Model` / `ModelModule` |
| `expert_hidden_dim` | `model_hidden_dim` |
| `expert_activation` | `model_activation` |
| `expert_norm` | `model_norm` |
| `expert_bias` | `model_bias` |
| `expert_dropout` | `model_dropout` |
| `expert_layer` | `model_module` |
| `MoE layer` | `ModelCoordinator` / `ModelCoordinatorLayer` |
| `MoE router` | `ModelRouter` / `ModelCoordinator` |
| `expert_routing` | `model_routing` / `model_selection` |
| `expert_utilization` | `model_utilization` |
| `dead_experts` | `inactive_models` |
| `expert_collapse` | `model_collapse` |
| `expert_specific_ratio` | `model_specific_ratio` |
| `expert_checkpoint` | `model_checkpoint` |

---

## Architecture Audit Summary

### Verdict
**The current implementation is a well-engineered conventional MoE system, but it is NOT a MoMMs system.** 

The fundamental architectural assumption — a single shared Transformer backbone with expert FFNs as MoE layers — is fundamentally incompatible with the MoMMs vision where each model is an independently trainable, independently checkpointable, complete model with its own embeddings, attention, FFN, and head.

### Effort Estimate
| Phase | Effort |
|-------|--------|
| Core architecture redesign | ~60% |
| Training pipeline redesign | ~20% |
| Data pipeline adaptation | ~10% |
| Configuration updates | ~5% |
| New components (registry, coordinator, fusion, GUI) | ~5% |

**Total: ~100% rewrite of core architecture, ~30% of total codebase**

---

## Recommendation

**Proceed with full MoMMs redesign** following the specification in the project requirements. The current MoE implementation, while well-engineered, cannot be incrementally adapted to MoMMs — the architectural assumptions are fundamentally different.

The audit identifies exactly what needs to be replaced vs. adapted. The data pipeline and evaluation modules can be largely reused with adaptations. The core model, training, and checkpointing systems need complete redesign.