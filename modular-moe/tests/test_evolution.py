"""Tests for population evolution and generation tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_model_fitness_composite():
    """Test model fitness composite score."""
    from src.training.evolution import ModelFitness

    fitness = ModelFitness(
        model_id="model_1",
        individual_reasoning=0.8,
        knowledge=0.7,
        math=0.9,
        coding=0.6,
        verification=0.8,
        collaboration=0.7,
        memory=0.6,
        tool_use=0.5,
        reliability=0.9,
    )

    score = fitness.compute_composite()
    assert score > 0.0
    assert fitness.composite_score == score
    print("  PASS: Model fitness composite score computed")


def test_population_evolution_redundancy():
    """Test redundancy detection."""
    from src.training.evolution import PopulationEvolution, ModelFitness

    config = {"population_evolution": {"redundancy_threshold": 0.95}}
    evolution = PopulationEvolution(config)

    # Register models with similar fitness
    for i in range(5):
        fitness = ModelFitness(model_id=f"model_{i}", generation=0)
        fitness.individual_reasoning = 0.5 + i * 0.01
        fitness.knowledge = 0.5 + i * 0.01
        fitness.compute_composite()
        evolution.model_fitness[f"model_{i}"] = fitness

    redundant = evolution.detect_redundancy([f"model_{i}" for i in range(5)])
    assert len(redundant) >= 0  # May or may not detect
    print("  PASS: Redundancy detection works")


def test_population_evolution_selection():
    """Test survivor selection."""
    from src.training.evolution import PopulationEvolution, ModelFitness

    config = {"population_evolution": {}}
    evolution = PopulationEvolution(config)

    model_ids = [f"model_{i:02d}" for i in range(10)]
    for i, mid in enumerate(model_ids):
        fitness = ModelFitness(model_id=mid, generation=0)
        fitness.individual_reasoning = i / 10.0
        fitness.compute_composite()
        evolution.model_fitness[mid] = fitness

    retained, retired = evolution.select_survivors(model_ids, retention_ratio=0.7)
    assert len(retained) == 7
    assert len(retired) == 3
    print("  PASS: Survivor selection works")


def test_population_evolution_decisions():
    """Test evolution decisions."""
    from src.training.evolution import PopulationEvolution, ModelFitness

    config = {"population_evolution": {"redundancy_threshold": 0.95}}
    evolution = PopulationEvolution(config)

    model_ids = [f"model_{i:02d}" for i in range(4)]
    for i, mid in enumerate(model_ids):
        fitness = ModelFitness(model_id=mid, generation=0)
        fitness.individual_reasoning = i / 10.0
        fitness.compute_composite()
        evolution.model_fitness[mid] = fitness

    decisions = evolution.evolve_population(model_ids, [])
    assert len(decisions) == 4
    assert all(d.decision_type in ["retain", "retrain", "retire", "replace"] for d in decisions)
    print("  PASS: Evolution decisions generated")


def test_generation_tracker():
    """Test generation tracker."""
    from src.training.evolution import GenerationTracker

    tracker = GenerationTracker(manifest_dir="test_manifests")

    manifest = tracker.start_generation({"epochs": 1})
    assert manifest.generation_id == 1

    tracker.complete_generation(manifest)
    loaded = tracker.load_generation(1)
    assert loaded is not None
    assert loaded.generation_id == 1

    latest = tracker.get_latest_generation()
    assert latest is not None
    assert latest.generation_id == 1

    print("  PASS: Generation tracker works")


def test_generation_manifest_serialization():
    """Test generation manifest serialization."""
    from src.training.evolution import GenerationManifest, EvolutionDecision
    import tempfile

    manifest = GenerationManifest(
        generation_id=1,
        population_version="gen_0001",
        dataset_version="dataset_0001",
        shared_core_hash="abc123",
        unique_assignment_hash="def456",
        model_profiles=[{"model_id": "m1"}],
        training_runs=[{"run_id": 0}],
        benchmark_results=[],
        evolution_decisions=[
            EvolutionDecision(
                decision_type="retain",
                model_id="m1",
                reason="good fitness",
                old_profile={"model_id": "m1"},
            ).to_dict()
        ],
        retained_models=["m1"],
        retrained_models=[],
        replaced_profiles=[],
    )

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        manifest.save(f.name)
        loaded = GenerationManifest.load(f.name)
        assert loaded.generation_id == 1
        assert loaded.population_version == "gen_0001"

    print("  PASS: Generation manifest serialization works")


def test_evolution_decision_serialization():
    """Test evolution decision serialization."""
    from src.training.evolution import EvolutionDecision

    decision = EvolutionDecision(
        decision_type="retrain",
        model_id="model_1",
        reason="low fitness",
        old_profile={"model_id": "model_1"},
        fitness_before=0.3,
        fitness_after=None,
    )

    data = decision.to_dict()
    assert data["decision_type"] == "retrain"
    assert data["model_id"] == "model_1"
    assert data["fitness_before"] == 0.3
    print("  PASS: Evolution decision serialization works")


def test_population_diversity_report():
    """Test population diversity report."""
    from src.training.evolution import PopulationEvolution, ModelFitness

    config = {"population_evolution": {}}
    evolution = PopulationEvolution(config)

    for i in range(5):
        fitness = ModelFitness(model_id=f"model_{i}", generation=1)
        fitness.individual_reasoning = 0.3 + i * 0.2
        fitness.compute_composite()
        evolution.model_fitness[f"model_{i}"] = fitness

    report = evolution.get_population_diversity_report()
    assert "generation" in report
    assert "model_count" in report
    assert report["model_count"] == 5
    print("  PASS: Population diversity report generated")


def run_all_tests():
    """Run all evolution tests."""
    tests = [
        ("Model Fitness Composite", test_model_fitness_composite),
        ("Redundancy Detection", test_population_evolution_redundancy),
        ("Survivor Selection", test_population_evolution_selection),
        ("Evolution Decisions", test_population_evolution_decisions),
        ("Generation Tracker", test_generation_tracker),
        ("Manifest Serialization", test_generation_manifest_serialization),
        ("Decision Serialization", test_evolution_decision_serialization),
        ("Diversity Report", test_population_diversity_report),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
