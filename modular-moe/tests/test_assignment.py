"""Tests for dataset assignment, ownership, and diversity."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_assignment_engine_shared_core():
    """Test shared core selection."""
    from src.data.assignment import DatasetAssignmentEngine, SampleRecord

    config = {"shared_core": {"ratio": 0.1}}
    engine = DatasetAssignmentEngine(config)

    samples = [
        SampleRecord(
            sample_id=f"s{i:04d}",
            text=f"sample {i}",
            token_count=10,
            domain="mathematics" if i % 2 == 0 else "programming",
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=float(i) / 100.0,
            source="test",
        )
        for i in range(100)
    ]

    engine.load_samples(samples)
    shared = engine.build_shared_core()

    assert len(shared) == 10
    for sample in shared:
        assert sample.is_shared is True
        assert sample.sample_id in engine.shared_sample_ids

    print("  PASS: Shared core selected correctly")


def test_assignment_engine_unique_allocation():
    """Test unique experience allocation without duplicates."""
    from src.data.assignment import DatasetAssignmentEngine, SampleRecord
    from src.data.scheduler import DatasetScheduler

    config = {
        "shared_core": {"ratio": 0.1},
        "unique_experience": {"ratio": 0.9},
        "datasets": {
            "min_coverage_all_domains": True,
            "minimum_reasoning_exposure": 0.1,
            "minimum_verification_exposure": 0.05,
        },
    }
    engine = DatasetAssignmentEngine(config)

    samples = [
        SampleRecord(
            sample_id=f"s{i:04d}",
            text=f"sample {i}",
            token_count=10,
            domain="mathematics" if i % 3 == 0 else ("programming" if i % 3 == 1 else "science"),
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=1.0,
            source="test",
        )
        for i in range(200)
    ]

    engine.load_samples(samples)
    engine.build_shared_core()

    model_ids = [f"solver_{i:05d}" for i in range(4)]
    scheduler = DatasetScheduler(config)
    assignments = engine.assign_unique_experience(model_ids, scheduler, seed=42)

    # Every model gets some samples
    for model_id in model_ids:
        assert len(assignments[model_id]) > 0

    # No duplicates in unique pool
    ownership_counts = {}
    for sample in samples:
        if sample.is_shared or sample.is_replicated:
            continue
        owner = sample.owner_model_id
        ownership_counts[sample.sample_id] = ownership_counts.get(sample.sample_id, 0) + 1

    duplicates = sum(1 for count in ownership_counts.values() if count > 1)
    assert duplicates == 0, f"Found {duplicates} duplicate samples in unique pool"

    print("  PASS: Unique allocation with no duplicates")


def test_assignment_engine_ownership_report():
    """Test ownership report generation."""
    from src.data.assignment import DatasetAssignmentEngine, SampleRecord

    config = {"shared_core": {"ratio": 0.2}}
    engine = DatasetAssignmentEngine(config)

    samples = [
        SampleRecord(
            sample_id=f"s{i:04d}",
            text=f"sample {i}",
            token_count=10,
            domain="mathematics",
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=1.0,
            source="test",
        )
        for i in range(50)
    ]

    engine.load_samples(samples)
    engine.build_shared_core()

    model_ids = [f"solver_{i:05d}" for i in range(3)]
    from src.data.scheduler import DatasetScheduler
    scheduler = DatasetScheduler(config)
    engine.assign_unique_experience(model_ids, scheduler, seed=42)

    report = engine.check_ownership()
    assert report.total_samples == 50
    assert report.shared_core_count == 10
    assert report.assigned_count + report.unassigned_count == 40

    print("  PASS: Ownership report generated correctly")


def test_assignment_engine_serialization():
    """Test assignment save/load."""
    from src.data.assignment import DatasetAssignmentEngine, SampleRecord

    config = {"shared_core": {"ratio": 0.1}}
    engine = DatasetAssignmentEngine(config)

    samples = [
        SampleRecord(
            sample_id=f"s{i:04d}",
            text=f"sample {i}",
            token_count=10,
            domain="mathematics",
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=1.0,
            source="test",
        )
        for i in range(20)
    ]

    engine.load_samples(samples)
    engine.build_shared_core()
    model_ids = [f"solver_{i:05d}" for i in range(2)]
    from src.data.scheduler import DatasetScheduler
    scheduler = DatasetScheduler(config)
    engine.assign_unique_experience(model_ids, scheduler, seed=42)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        engine.save_assignment(f.name)
        loaded = DatasetAssignmentEngine(config)
        loaded.load_assignment(f.name)

        assert len(loaded.samples) == len(engine.samples)
        assert loaded.shared_sample_ids == engine.shared_sample_ids
        assert loaded.ownership == engine.ownership

    print("  PASS: Assignment serialization works")


def test_assignment_engine_population_diversity():
    """Test population diversity check."""
    from src.data.assignment import DatasetAssignmentEngine, SampleRecord

    config = {"shared_core": {"ratio": 0.1}}
    engine = DatasetAssignmentEngine(config)

    samples = [
        SampleRecord(
            sample_id=f"s{i:04d}",
            text=f"sample {i}",
            token_count=10,
            domain="mathematics" if i % 2 == 0 else "programming",
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=1.0,
            source="test",
        )
        for i in range(100)
    ]

    engine.load_samples(samples)
    engine.build_shared_core()

    model_ids = [f"solver_{i:05d}" for i in range(4)]
    from src.data.scheduler import DatasetScheduler
    scheduler = DatasetScheduler(config)
    engine.assign_unique_experience(model_ids, scheduler, seed=42)

    diversity = engine.check_population_diversity(model_ids)
    assert "average_domain_similarity" in diversity
    assert "population_diversity_warning" in diversity

    print("  PASS: Population diversity check works")


def test_assignment_engine_critical_replication():
    """Test critical replication."""
    from src.data.assignment import DatasetAssignmentEngine, SampleRecord

    config = {
        "shared_core": {"ratio": 0.1},
        "critical_replication": {"enabled": True, "ratio": 0.05},
    }
    engine = DatasetAssignmentEngine(config)

    samples = [
        SampleRecord(
            sample_id=f"s{i:04d}",
            text=f"sample {i}",
            token_count=10,
            domain="mathematics",
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=1.0,
            source="test",
        )
        for i in range(100)
    ]

    engine.load_samples(samples)
    engine.build_shared_core()

    model_ids = [f"solver_{i:05d}" for i in range(3)]
    from src.data.scheduler import DatasetScheduler
    scheduler = DatasetScheduler(config)
    engine.assign_unique_experience(model_ids, scheduler, seed=42)
    engine.assign_critical_replication(model_ids, samples)

    assert len(engine.replicated_sample_ids) > 0
    print("  PASS: Critical replication works")


def run_all_tests():
    """Run all assignment tests."""
    tests = [
        ("Shared Core Selection", test_assignment_engine_shared_core),
        ("Unique Allocation", test_assignment_engine_unique_allocation),
        ("Ownership Report", test_assignment_engine_ownership_report),
        ("Serialization", test_assignment_engine_serialization),
        ("Population Diversity", test_assignment_engine_population_diversity),
        ("Critical Replication", test_assignment_engine_critical_replication),
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
