"""Tests for hard-example mining and curriculum update."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_hard_example_miner_mines_failure():
    """Test that hard examples are mined from population failure."""
    from src.training.hard_examples import HardExampleMiner

    config = {
        "hard_example_mining": {
            "disagreement_threshold": 0.3,
            "confidence_threshold": 0.5,
            "min_failure_count": 2,
        }
    }
    miner = HardExampleMiner(config)

    candidates = {
        "model_a": {"text": "answer A", "confidence": 0.3},
        "model_b": {"text": "answer B", "confidence": 0.4},
    }
    verifications = {
        "model_a": {"passed": False, "confidence": 0.3},
        "model_b": {"passed": False, "confidence": 0.4},
    }

    example = miner.mine_from_population(
        task_id="task_1",
        prompt="Solve this hard math problem",
        candidates=candidates,
        verifications=verifications,
    )

    assert example is not None
    assert len(example.wrong_candidates) == 2
    assert example.difficulty in ["D2", "D3", "D4", "D5"]
    print("  PASS: Hard example mined from failure")


def test_hard_example_miner_skips_easy():
    """Test that easy tasks are not mined as hard examples."""
    from src.training.hard_examples import HardExampleMiner

    config = {
        "hard_example_mining": {
            "confidence_threshold": 0.5,
            "min_failure_count": 2,
        }
    }
    miner = HardExampleMiner(config)

    candidates = {
        "model_a": {"text": "correct answer", "confidence": 0.9},
        "model_b": {"text": "correct answer", "confidence": 0.95},
    }
    verifications = {
        "model_a": {"passed": True, "confidence": 0.9},
        "model_b": {"passed": True, "confidence": 0.95},
    }

    example = miner.mine_from_population(
        task_id="task_easy",
        prompt="What is 2+2?",
        candidates=candidates,
        verifications=verifications,
    )

    assert example is None
    print("  PASS: Easy task skipped")


def test_failure_logging():
    """Test failure logging."""
    from src.training.hard_examples import HardExampleMiner, FailureCategory

    config = {"hard_example_mining": {}}
    miner = HardExampleMiner(config)

    failure = miner.log_failure(
        model_id="model_1",
        task_id="task_1",
        category=FailureCategory.MATH_ERROR.value,
        description="Incorrect calculation",
        loss=5.0,
        confidence=0.3,
    )

    assert failure.model_id == "model_1"
    assert failure.category == "math_error"
    assert len(miner.failure_logs) == 1
    print("  PASS: Failure logged")


def test_failure_stats():
    """Test failure statistics."""
    from src.training.hard_examples import HardExampleMiner, FailureCategory

    config = {"hard_example_mining": {}}
    miner = HardExampleMiner(config)

    miner.log_failure("model_1", "t1", FailureCategory.MATH_ERROR.value, "bad math")
    miner.log_failure("model_2", "t2", FailureCategory.CODING_ERROR.value, "bad code")
    miner.log_failure("model_1", "t3", FailureCategory.MATH_ERROR.value, "bad math")

    stats = miner.get_failure_stats()
    assert stats["total_failures"] == 3
    assert "math_error" in stats["category_distribution"]
    assert "coding_error" in stats["category_distribution"]
    print("  PASS: Failure stats computed")


def test_hard_example_serialization():
    """Test hard example save/load."""
    from src.training.hard_examples import HardExampleMiner
    import tempfile

    config = {"hard_example_mining": {}}
    miner = HardExampleMiner(config)

    candidates = {
        "model_a": {"text": "wrong", "confidence": 0.3},
    }
    verifications = {
        "model_a": {"passed": False, "confidence": 0.3},
    }
    miner.mine_from_population("t1", "hard problem", candidates, verifications)
    miner.log_failure("model_a", "t1", "reasoning_error", "bad reasoning")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        miner.save(f.name)
        loaded = HardExampleMiner(config)
        loaded.load(f.name)

        assert len(loaded.hard_examples) == len(miner.hard_examples)
        assert len(loaded.failure_logs) == len(miner.failure_logs)

    print("  PASS: Hard example serialization works")


def test_curriculum_updater():
    """Test curriculum updater."""
    from src.training.hard_examples import (
        CurriculumUpdater,
        HardExample,
        DifficultyLevel,
    )
    from src.data.scheduler import ModelMixture

    config = {"hard_example_mining": {}}
    updater = CurriculumUpdater(config)

    mixture = ModelMixture(
        model_id="solver_0001",
        domain_weights={"mathematics": 0.2, "programming": 0.2, "reasoning": 0.2, "science": 0.2, "general": 0.2},
        difficulty_weights={"D2": 0.5, "D3": 0.3, "D4": 0.2},
        task_weights={"reasoning": 1.0},
        language_weights={"en": 1.0},
    )
    mixture.normalize()

    hard_examples = [
        HardExample(
            example_id="h1",
            task_id="t1",
            prompt="hard math problem",
            wrong_candidates=["wrong"],
            correct_candidate="right",
            critiques=[],
            verifications=[],
            failure_categories=["math_error"],
            difficulty=DifficultyLevel.D4.value,
            domain="mathematics",
            task_type="reasoning",
            model_ids=["model_a"],
            confidence_scores=[0.3],
        ),
    ]

    updated = updater.update_mixture(mixture, hard_examples, {})

    assert "mathematics" in updated.domain_weights
    assert updated.domain_weights["mathematics"] > 0.2
    assert updated.difficulty_weights["D4"] > 0.2
    print("  PASS: Curriculum updated based on hard examples")


def test_difficulty_levels():
    """Test difficulty level enum."""
    from src.training.hard_examples import DifficultyLevel

    assert DifficultyLevel.D0.value == "D0"
    assert DifficultyLevel.D5.value == "D5"
    print("  PASS: Difficulty levels defined")


def test_failure_categories():
    """Test failure categories."""
    from src.training.hard_examples import FailureCategory

    assert FailureCategory.MATH_ERROR.value == "math_error"
    assert FailureCategory.HALLUCINATION.value == "hallucination"
    assert FailureCategory.VERIFICATION_FAILURE.value == "verification_failure"
    print("  PASS: Failure categories defined")


def run_all_tests():
    """Run all hard-example tests."""
    tests = [
        ("Mine Failure", test_hard_example_miner_mines_failure),
        ("Skip Easy", test_hard_example_miner_skips_easy),
        ("Failure Logging", test_failure_logging),
        ("Failure Stats", test_failure_stats),
        ("Serialization", test_hard_example_serialization),
        ("Curriculum Updater", test_curriculum_updater),
        ("Difficulty Levels", test_difficulty_levels),
        ("Failure Categories", test_failure_categories),
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
