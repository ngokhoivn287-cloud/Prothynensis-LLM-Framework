"""Tests for MrTP population-selection engine and multimodal extensions."""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np

from src.training.population_selection import (
    PopulationSelectionEngine,
    CandidateRecord,
    CandidateStatus,
    CapabilityVector,
)
from src.training.evolution import ModelFitness
from src.model.vision_model import VisionModelConfig, VisionReasoningModel, create_vision_model, count_vision_model_parameters
from src.model.computer_use_model import (
    ComputerUseModelConfig,
    ComputerUseModel,
    create_computer_use_model,
    count_computer_use_model_parameters,
)
from src.data.multimodal_schema import (
    MultimodalSample,
    Modality,
    DatasetProfile,
    ModalityAdapter,
    TextAdapter,
    ImageAdapter,
    VideoAdapter,
    ScreenAdapter,
    MultimodalDatasetRegistry,
)
from src.utils.config import load_config
from src.utils.paths import get_temp_root, get_checkpoints_root


def test_capability_vector_cosine_similarity():
    a = CapabilityVector(reasoning=0.9, mathematics=0.8, language=0.7, computer_vision=0.1)
    b = CapabilityVector(reasoning=0.9, mathematics=0.8, language=0.7, computer_vision=0.1)
    assert a.cosine_similarity(b) == pytest.approx(1.0, abs=0.01)
    c = CapabilityVector(reasoning=0.1, mathematics=0.1, language=0.1, computer_vision=0.9)
    sim = a.cosine_similarity(c)
    assert sim < 0.5


def test_candidate_record_composite_score():
    c = CandidateRecord(model_id="test", overall_quality=0.9, reasoning_score=0.85, verification_score=0.8)
    c.compute_composite()
    assert 0.0 < c.composite_score <= 1.0


def test_population_selection_retains_top():
    engine = PopulationSelectionEngine({"quality_floor": 0.0, "redundancy_threshold": 0.99, "retention_ratio": 0.5})
    candidates = [
        CandidateRecord(model_id=f"c{i}", model_family="general_solver",
                        overall_quality=0.9 - i * 0.1, reasoning_score=0.8 - i * 0.1,
                        verification_score=0.7 - i * 0.1)
        for i in range(10)
    ]
    for c in candidates:
        c.capability_vector = CapabilityVector(reasoning=c.overall_quality, mathematics=c.overall_quality,
                                               language=c.overall_quality)
    retained, discarded = engine.select(candidates)
    assert len(retained) >= 1
    assert all(r.status == CandidateStatus.RETAINED for r in retained)
    assert all(d.status == CandidateStatus.DISCARDED for d in discarded)


def test_population_selection_redundancy_penalty():
    engine = PopulationSelectionEngine({"quality_floor": 0.0, "redundancy_threshold": 0.90, "retention_ratio": 0.5})
    cap = CapabilityVector(reasoning=0.9, mathematics=0.9, language=0.9, computer_vision=0.1)
    candidates = [
        CandidateRecord(model_id=f"c{i}", model_family="general_solver", overall_quality=0.8,
                        capability_vector=cap)
        for i in range(10)
    ]
    retained, discarded = engine.select(candidates)
    assert len(discarded) > 0, "Expected redundancy to trigger discards"


def test_population_selection_diversity_scoring():
    engine = PopulationSelectionEngine({"quality_floor": 0.0})
    diverse = CapabilityVector(reasoning=0.9, mathematics=0.1, language=0.1)
    narrow = CapabilityVector(reasoning=0.95, mathematics=0.9, language=0.9)
    c1 = CandidateRecord(model_id="diverse", model_family="general_solver", overall_quality=0.8,
                         capability_vector=diverse)
    c2 = CandidateRecord(model_id="narrow", model_family="general_solver", overall_quality=0.8,
                         capability_vector=narrow)
    engine.compute_diversity_scores([c1, c2])
    assert c1.diversity >= 0.0
    assert c2.diversity >= 0.0


def test_population_selection_hierarchy_build():
    engine = PopulationSelectionEngine({"quality_floor": 0.0})
    candidates = [
        CandidateRecord(model_id=f"c{i}", model_family="general_solver", overall_quality=0.9 - i * 0.1,
                        reasoning_score=0.8, verification_score=0.7)
        for i in range(5)
    ]
    retained, _ = engine.select(candidates)
    tiers = engine.build_hierarchy(retained, {"ultimate": 1, "master": 1, "chief": 1, "orchestral": 1})
    assert sum(len(v) for v in tiers.values()) <= len(retained)


def test_population_selection_quality_floor():
    engine = PopulationSelectionEngine({"quality_floor": 0.8, "retention_ratio": 1.0})
    candidates = [
        CandidateRecord(model_id=f"c{i}", model_family="general_solver", overall_quality=0.9 - i * 0.2)
        for i in range(5)
    ]
    retained, discarded = engine.select(candidates)
    assert all(r.overall_quality >= 0.8 for r in retained)
    assert all(d.overall_quality < 0.8 for d in discarded)


def test_vision_model_forward_pass():
    config = VisionModelConfig(model_id="test_vision", vocab_size=1024, max_seq_len=128, hidden_dim=256,
                               num_layers=2, num_heads=4, head_dim=64, ffn_hidden_dim=1024,
                               vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
                               image_size=64, patch_size=8)
    model = create_vision_model(config)
    dummy_image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        logits = model(dummy_image)
    assert logits.shape[0] == 1


def test_vision_model_parameter_count():
    config = VisionModelConfig(model_id="test_vision", vocab_size=50304, max_seq_len=2048, hidden_dim=512,
                               num_layers=12, num_heads=8, head_dim=64, ffn_hidden_dim=2048)
    model = create_vision_model(config)
    counts = count_vision_model_parameters(model)
    assert counts["total"] > 0


def test_computer_use_model_forward_pass():
    config = ComputerUseModelConfig(model_id="test_cu", vocab_size=1024, max_seq_len=128, hidden_dim=256,
                                    num_layers=2, num_heads=4, head_dim=64, ffn_hidden_dim=1024,
                                    vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
                                    image_size=64, patch_size=8)
    model = create_computer_use_model(config)
    dummy_image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        outputs = model(dummy_image)
    assert "logits" in outputs
    assert "actions" in outputs


def test_computer_use_model_parameter_count():
    config = ComputerUseModelConfig(model_id="test_cu", vocab_size=50304, max_seq_len=2048, hidden_dim=512,
                                    num_layers=12, num_heads=8, head_dim=64, ffn_hidden_dim=2048)
    model = create_computer_use_model(config)
    counts = count_computer_use_model_parameters(model)
    assert counts["total"] > 0


def test_multimodal_sample_creation():
    sample = MultimodalSample(
        sample_id="test_1",
        modality=Modality.IMAGE,
        domain="computer_vision",
        task="image_classification",
        difficulty="D2",
        quality=0.9,
    )
    assert sample.modality == Modality.IMAGE
    d = sample.to_dict()
    assert d["modality"] == "image"


def test_multimodal_dataset_registry():
    registry = MultimodalDatasetRegistry()
    registry.register_profile(DatasetProfile(
        name="imagenet_test", domain="computer_vision", difficulty="D3",
        task_type="image_classification", modality=Modality.IMAGE,
    ))
    profiles = registry.list_by_modality(Modality.IMAGE)
    assert len(profiles) == 1
    assert profiles[0].name == "imagenet_test"


def test_text_adapter():
    adapter = TextAdapter()
    sample = MultimodalSample(
        sample_id="t1", modality=Modality.TEXT, domain="language", task="generation", difficulty="D1",
        quality=0.9, payload=b"hello world",
    )
    result = adapter.preprocess(sample)
    assert result is not None
    assert "text" in result


def test_image_adapter():
    adapter = ImageAdapter()
    sample = MultimodalSample(
        sample_id="i1", modality=Modality.IMAGE, domain="computer_vision", task="classification", difficulty="D2",
        quality=0.9,
    )
    result = adapter.preprocess(sample)
    assert result is None


def test_screen_adapter():
    adapter = ScreenAdapter()
    assert adapter.modality == Modality.SCREEN


def test_mrtp_configs_exist():
    for name in ["mini3_mrtp", "pro3_mrtp", "ultra3_mrtp", "trinity3_mrtp"]:
        path = Path(__file__).parent.parent / "configs" / f"{name}.yaml"
        assert path.exists(), f"Missing config: {path}"


def test_mrtp_config_population_selection_section():
    for name in ["mini3_mrtp", "pro3_mrtp", "ultra3_mrtp", "trinity3_mrtp"]:
        config = load_config(f"configs/{name}.yaml")
        assert "population_selection" in config
        ps = config["population_selection"]
        assert "redundancy_threshold" in ps
        assert "retention_ratio" in ps
        assert "quality_floor" in ps


def test_mrtp_config_parameter_accounting():
    for name in ["mini3_mrtp", "pro3_mrtp", "ultra3_mrtp", "trinity3_mrtp"]:
        config = load_config(f"configs/{name}.yaml")
        assert "solver" in config
        assert "vision_population" in config or "computer_use_population" in config or name in ("mini3_mrtp", "pro3_mrtp")
        if "system" in config:
            assert "solver_params" in config["system"]
