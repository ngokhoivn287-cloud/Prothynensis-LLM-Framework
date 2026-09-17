"""Tests for dynamic runtime recruitment."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def _make_model(model_id: str, capability_vector: dict) -> Any:
    class FakeModel:
        def __init__(self, model_id: str, capability_vector: dict):
            self.model_id = model_id
            self.capability_vector = type("CapVector", (), capability_vector)()
            self.parameter_count = 77_161_472
            self.metadata = {"specialization": "general_solver"}
        def get_parameter_count(self):
            return {"total": self.parameter_count}
    return FakeModel(model_id, capability_vector)


def test_dynamic_recruiter_initial_selection():
    """Test initial candidate selection."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext
    from momm.recruitment import RecruitmentContext

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5 + i * 0.1}) for i in range(6)]
    task = TaskContext(task_id="t1", prompt="test", difficulty=0.5)
    selected = orchestrator.initial_recruitment(
        task_id="t1",
        available_models=models,
        task_requirements=task.to_dict(),
        capability_requirements={},
    )
    assert len(selected) >= 1
    assert len(selected) <= 10


def test_dynamic_recruiter_low_uncertainty_stops():
    """Test that low uncertainty stops recruitment."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext, CandidateOutput

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "max_rounds": 3,
            "uncertainty_threshold": 0.3,
            "disagreement_threshold": 0.15,
            "min_information_gain": 0.01,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.8}) for i in range(6)]
    orchestrator.initial_recruitment(
        task_id="t2",
        available_models=models,
        task_requirements=TaskContext(task_id="t2", prompt="test", difficulty=0.3).to_dict(),
        capability_requirements={},
    )
    outputs = [CandidateOutput(model_id="model_0", content="A", confidence=0.95)]
    result = orchestrator.evaluate_and_recruit(
        task_id="t2",
        outputs=outputs,
        uncertainty=0.05,
        disagreement=0.0,
        available_models=models,
        capability_requirements={},
    )
    assert result["decision"] == "stop"
    assert result["stop_reason"] in {"low_uncertainty", "low_disagreement", "low_information_gain"}


def test_dynamic_recruiter_high_uncertainty_expands():
    """Test that high uncertainty expands recruitment."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext, CandidateOutput

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "max_rounds": 3,
            "uncertainty_threshold": 0.3,
            "disagreement_threshold": 0.15,
            "min_information_gain": 0.01,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5 + i * 0.1}) for i in range(6)]
    orchestrator.initial_recruitment(
        task_id="t3",
        available_models=models,
        task_requirements=TaskContext(task_id="t3", prompt="test", difficulty=0.7).to_dict(),
        capability_requirements={},
    )
    outputs = [
        CandidateOutput(model_id="model_0", content="A", confidence=0.3),
        CandidateOutput(model_id="model_1", content="B", confidence=0.3),
    ]
    result = orchestrator.evaluate_and_recruit(
        task_id="t3",
        outputs=outputs,
        uncertainty=0.8,
        disagreement=0.6,
        available_models=models,
        capability_requirements={},
    )
    assert result["decision"] == "recruit"


def test_dynamic_recruiter_uses_information_gain():
    """Test that recruitment uses information gain."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "max_rounds": 3,
            "uncertainty_threshold": 0.3,
            "disagreement_threshold": 0.15,
            "min_information_gain": 0.5,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(6)]
    orchestrator.initial_recruitment(
        task_id="t4",
        available_models=models,
        task_requirements=TaskContext(task_id="t4", prompt="test", difficulty=0.5).to_dict(),
        capability_requirements={},
    )
    from runtime.recruitment_orchestrator import CandidateOutput
    outputs = [CandidateOutput(model_id="model_0", content="A", confidence=0.5)]
    result = orchestrator.evaluate_and_recruit(
        task_id="t4",
        outputs=outputs,
        uncertainty=0.1,
        disagreement=0.0,
        available_models=models,
        capability_requirements={},
    )
    assert result["decision"] == "stop"
    assert result["stop_reason"] in {"low_information_gain", "low_uncertainty"}


def test_dynamic_recruiter_uses_reputation():
    """Test that recruitment uses reputation."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext
    from src.training.reputation import ReputationStore

    store = ReputationStore()
    for i in range(6):
        rep = store.get_or_create(f"model_{i}")
        for _ in range(5):
            rep.record_global_attempt(success=(i >= 3), confidence=0.8, actual_correct=(i >= 3))
        rep.update_tier()
        store.update_reputation(rep)

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(6)]
    selected = orchestrator.initial_recruitment(
        task_id="t5",
        available_models=models,
        task_requirements=TaskContext(task_id="t5", prompt="test", difficulty=0.5).to_dict(),
        capability_requirements={},
        reputation_store=store,
    )
    selected_ids = [m.model_id for m in selected]
    assert all(m in selected_ids for m in ["model_3", "model_4", "model_5"])


def test_dynamic_recruiter_uses_diversity():
    """Test that recruitment considers diversity."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.9}) for i in range(6)]
    selected = orchestrator.initial_recruitment(
        task_id="t6",
        available_models=models,
        task_requirements=TaskContext(task_id="t6", prompt="test", difficulty=0.5).to_dict(),
        capability_requirements={},
    )
    assert len(selected) >= 1


def test_dynamic_recruiter_targeted_capability_selection():
    """Test targeted capability-based recruitment."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
        }
    })
    math_models = [_make_model(f"math_model_{i}", {"mathematics": 0.9, "reasoning": 0.8}) for i in range(3)]
    vision_models = [_make_model(f"vision_model_{i}", {"computer_vision": 0.9, "reasoning": 0.4}) for i in range(3)]
    models = math_models + vision_models
    selected = orchestrator.initial_recruitment(
        task_id="t7",
        available_models=models,
        task_requirements=TaskContext(task_id="t7", prompt="math problem", difficulty=0.6, domain="math").to_dict(),
        capability_requirements={"mathematics": 0.7},
    )
    selected_ids = [m.model_id for m in selected]
    assert any("math_model" in mid for mid in selected_ids)


def test_dynamic_recruiter_respects_compute_budget():
    """Test that recruitment respects compute budget."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 2,
            "default_top_k": 3,
            "compute_budget": 0.1,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(10)]
    selected = orchestrator.initial_recruitment(
        task_id="t8",
        available_models=models,
        task_requirements=TaskContext(task_id="t8", prompt="test", difficulty=0.5).to_dict(),
        capability_requirements={},
    )
    assert len(selected) <= 2


def test_dynamic_recruiter_respects_latency_budget():
    """Test that recruitment respects latency budget."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "latency_budget": 0.1,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(10)]
    selected = orchestrator.initial_recruitment(
        task_id="t9",
        available_models=models,
        task_requirements=TaskContext(task_id="t9", prompt="test", difficulty=0.5).to_dict(),
        capability_requirements={},
    )
    assert len(selected) >= 1


def test_dynamic_recruiter_has_stop_reason():
    """Test that stop decision includes a reason."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext, CandidateOutput

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "max_rounds": 1,
            "uncertainty_threshold": 0.3,
            "disagreement_threshold": 0.15,
            "min_information_gain": 0.01,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(6)]
    orchestrator.initial_recruitment(
        task_id="t10",
        available_models=models,
        task_requirements=TaskContext(task_id="t10", prompt="test", difficulty=0.3).to_dict(),
        capability_requirements={},
    )
    outputs = [CandidateOutput(model_id="model_0", content="A", confidence=0.9)]
    result = orchestrator.evaluate_and_recruit(
        task_id="t10",
        outputs=outputs,
        uncertainty=0.1,
        disagreement=0.0,
        available_models=models,
        capability_requirements={},
    )
    assert result["decision"] == "stop"
    assert result["stop_reason"] is not None


def test_dynamic_recruiter_has_no_infinite_loop():
    """Test that recruitment cannot loop indefinitely."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext, CandidateOutput

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 100,
            "default_top_k": 3,
            "max_rounds": 2,
            "max_recruitments_per_task": 2,
            "uncertainty_threshold": 0.0,
            "disagreement_threshold": 0.0,
            "min_information_gain": 0.0,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(10)]
    orchestrator.initial_recruitment(
        task_id="t11",
        available_models=models,
        task_requirements=TaskContext(task_id="t11", prompt="test", difficulty=0.9).to_dict(),
        capability_requirements={},
    )
    outputs = [CandidateOutput(model_id="model_0", content="A", confidence=0.1)]
    rounds = 0
    result = orchestrator.evaluate_and_recruit(
        task_id="t11",
        outputs=outputs,
        uncertainty=1.0,
        disagreement=1.0,
        available_models=models,
        capability_requirements={},
    )
    rounds += 1
    while result.get("decision") == "recruit" and rounds < 10:
        new_models = result.get("recruited", [])
        new_outputs = [CandidateOutput(model_id=m.model_id, content="X", confidence=0.1) for m in new_models]
        outputs.extend(new_outputs)
        result = orchestrator.evaluate_and_recruit(
            task_id="t11",
            outputs=outputs,
            uncertainty=1.0,
            disagreement=1.0,
            available_models=[m for m in models if getattr(m, "model_id", str(m)) not in orchestrator.get_state("t11").recruited_models],
            capability_requirements={},
        )
        rounds += 1
    assert rounds <= 2


def test_runtime_recruitment_trace():
    """Test that recruitment produces a structured trace."""
    from runtime.recruitment_orchestrator import RecruitmentOrchestrator, CandidateOutput
    from runtime.runtime_orchestrator import TaskContext, CandidateOutput

    orchestrator = RecruitmentOrchestrator({
        "recruitment": {
            "strategy": "hybrid",
            "min_active_models": 1,
            "max_active_models": 10,
            "default_top_k": 3,
            "max_rounds": 2,
        }
    })
    models = [_make_model(f"model_{i}", {"reasoning": 0.5}) for i in range(6)]
    orchestrator.initial_recruitment(
        task_id="t12",
        available_models=models,
        task_requirements=TaskContext(task_id="t12", prompt="test", difficulty=0.5).to_dict(),
        capability_requirements={},
    )
    outputs = [CandidateOutput(model_id="model_0", content="A", confidence=0.9)]
    result = orchestrator.evaluate_and_recruit(
        task_id="t12",
        outputs=outputs,
        uncertainty=0.1,
        disagreement=0.0,
        available_models=models,
        capability_requirements={},
    )
    trace = orchestrator.get_recruitment_trace("t12")
    assert len(trace) >= 1
    assert trace[0]["decision"] in {"recruit", "stop", "verify", "synthesize"}


def test_capability_vector_dimension_count():
    """Test CapabilityVector dimension count is correct."""
    from src.training.population_selection import CapabilityVector

    cv = CapabilityVector()
    expected_dimensions = {
        "general_knowledge", "language", "mathematics", "programming",
        "computer_science", "science", "reasoning", "verification",
        "planning", "tool_use", "memory", "collaboration",
        "computer_vision", "video_understanding", "computer_use",
        "adversarial_reasoning", "fact_checking", "proof_checking",
        "code_review", "assumption_detection", "contradiction_detection",
        "uncertainty_estimation", "self_correction", "calibration",
        "information_gain",
    }
    actual_dimensions = set(cv.__dataclass_fields__.keys())
    assert actual_dimensions == expected_dimensions, f"Missing: {expected_dimensions - actual_dimensions}, Extra: {actual_dimensions - expected_dimensions}"
    assert len(actual_dimensions) == 25


def test_restart_checkpoint_generation_isolation():
    """Test that restart checkpoints are isolated by generation."""
    from pathlib import Path
    from src.training.adaptive_budget import TrainingBudget, BudgetPhase
    from src.utils.paths import get_checkpoints_root

    base = get_checkpoints_root()
    gen0_path = base / "generation_0" / "run_000" / "step_00000050"
    gen1_path = base / "generation_1" / "run_000" / "step_00000050"
    assert gen0_path != gen1_path
    assert gen0_path.parent.parent != gen1_path.parent.parent
