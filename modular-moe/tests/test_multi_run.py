"""Tests for multi-run training and recruitment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_run_metrics_composite_score():
    """Test composite score computation."""
    from src.training.multi_run import RunMetrics

    metrics = RunMetrics(
        run_id=0,
        train_loss=2.0,
        val_loss=2.5,
        reasoning_score=0.8,
        knowledge_score=0.7,
        domain_score=0.6,
        verification_score=0.9,
        stability_score=0.85,
        overfit_score=0.9,
    )

    score = metrics.compute_composite()
    assert score > 0.0
    assert metrics.composite_score == score
    print("  PASS: Composite score computed")


def test_multi_run_trainer_initialization():
    """Test multi-run trainer initialization."""
    from src.training.multi_run import MultiRunTrainer, RunSearchStrategy

    config = {
        "model": {"hidden_dim": 64, "num_layers": 2, "num_heads": 2, "head_dim": 32, "vocab_size": 100, "max_seq_len": 32},
        "training": {"seed": 42},
    }

    trainer = MultiRunTrainer(
        model_class=None,
        train_dataloader=None,
        eval_dataloader=None,
        config=config,
        output_dir="test_multi_run",
        num_runs=2,
    )

    assert trainer.num_runs == 2
    assert len(trainer.runs) == 0
    print("  PASS: Multi-run trainer initialized")


def test_run_search_strategy():
    """Test run search strategies."""
    from src.training.multi_run import RunSearchStrategy

    param_grid = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "batch_size": [4, 8],
    }

    grid_configs = RunSearchStrategy.grid_search(param_grid)
    assert len(grid_configs) == 6

    random_configs = RunSearchStrategy.random_search(param_grid, num_trials=3, seed=42)
    assert len(random_configs) == 3
    print("  PASS: Run search strategies work")


def test_activation_budget():
    """Test activation budget tracking."""
    from src.momm.recruitment import ActivationBudget

    budget = ActivationBudget(
        active_model_count=10,
        parameters_per_model=77_161_472,
        active_parameters=771_614_720,
    )

    assert budget.active_parameters_billions == 0.77161472
    assert budget.active_parameters_trillions == 0.00077161472
    print("  PASS: Activation budget tracks correctly")


def test_dynamic_recruiter():
    """Test dynamic recruiter."""
    from src.momm.recruitment import DynamicRecruiter, RecruitmentContext

    config = {
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "compute_budget": 1.0,
            "latency_budget": 60.0,
            "confidence_threshold": 0.85,
            "disagreement_threshold": 0.15,
        }
    }

    recruiter = DynamicRecruiter(config)

    # Create mock models
    class MockModel:
        def __init__(self, name):
            self.name = name

    available = [MockModel(f"model_{i}") for i in range(10)]

    # Easy task
    context = RecruitmentContext(task_difficulty=0.2, uncertainty=0.1)
    recruited = recruiter.recruit(context, available, [])
    assert len(recruited) >= 1
    assert len(recruited) <= 10

    # Hard task
    context = RecruitmentContext(task_difficulty=0.9, uncertainty=0.8)
    recruited = recruiter.recruit(context, available, [])
    assert len(recruited) >= 3
    print("  PASS: Dynamic recruiter works")


def test_recruitment_budget():
    """Test recruitment budget enforcement."""
    from momm.recruitment import RecruitmentBudget, ActivationBudget

    config = {
        "recruitment": {
            "max_active_models": 32,
            "compute_budget": 1.0,
        }
    }

    budget = RecruitmentBudget(config)
    activation = ActivationBudget(active_model_count=10, parameters_per_model=77_161_472)

    assert budget.can_activate(5, activation) is True
    assert budget.can_activate(25, activation) is False  # Would exceed max
    assert budget.get_max_activatable(activation) == 22
    print("  PASS: Recruitment budget enforces limits")


def test_run_metrics_serialization():
    """Test run metrics serialization."""
    from src.training.multi_run import RunMetrics

    metrics = RunMetrics(
        run_id=1,
        train_loss=1.5,
        val_loss=2.0,
        reasoning_score=0.8,
    )
    metrics.compute_composite()

    data = metrics.to_dict()
    assert data["run_id"] == 1
    assert data["train_loss"] == 1.5
    assert "composite_score" in data
    print("  PASS: Run metrics serialization works")


def run_all_tests():
    """Run all multi-run/recruitment tests."""
    tests = [
        ("Composite Score", test_run_metrics_composite_score),
        ("Multi-Run Trainer Init", test_multi_run_trainer_initialization),
        ("Run Search Strategy", test_run_search_strategy),
        ("Activation Budget", test_activation_budget),
        ("Dynamic Recruiter", test_dynamic_recruiter),
        ("Recruitment Budget", test_recruitment_budget),
        ("Run Metrics Serialization", test_run_metrics_serialization),
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
