"""Tests for self-reflection and collective deliberation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_self_assessment_basic():
    """Test basic self-assessment."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"self_reflection": {"enabled": True}})
    assessment = engine.self_assess(
        model_id="model_1",
        task_id="task_1",
        contribution="I believe the answer is A because of evidence X.",
        evidence=["evidence X", "evidence Y"],
        confidence=0.8,
    )
    assert assessment.model_id == "model_1"
    assert assessment.confidence == 0.8
    assert len(assessment.strengths) > 0


def test_self_assessment_detects_weaknesses():
    """Test that self-assessment detects weaknesses."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"self_reflection": {"enabled": True}})
    assessment = engine.self_assess(
        model_id="model_1",
        task_id="task_1",
        contribution="Certainly the answer is correct.",
        evidence=[],
        confidence=0.95,
    )
    assert len(assessment.weaknesses) > 0


def test_peer_challenge_basic():
    """Test basic peer challenge."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"peer_challenge": {"enabled": True}})
    assessment = engine.peer_challenge(
        reviewer_id="model_2",
        target_id="model_1",
        task_id="task_1",
        target_contribution="The answer is A.",
        reviewer_contribution="I found an error in your reasoning.",
    )
    assert assessment.reviewer_id == "model_2"
    assert assessment.target_id == "model_1"


def test_peer_challenge_detects_error():
    """Test that peer challenge detects caught errors."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"peer_challenge": {"enabled": True}})
    assessment = engine.peer_challenge(
        reviewer_id="model_2",
        target_id="model_1",
        task_id="task_1",
        target_contribution="The answer is A.",
        reviewer_contribution="I found an error in your reasoning.",
    )
    assert assessment.caught_missed_error is True


def test_peer_challenge_detects_overconfidence():
    """Test that peer challenge detects overconfidence."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"peer_challenge": {"enabled": True}})
    assessment = engine.peer_challenge(
        reviewer_id="model_2",
        target_id="model_1",
        task_id="task_1",
        target_contribution="Certainly definitely always correct.",
        reviewer_contribution="I disagree.",
    )
    assert assessment.confident_without_evidence is True


def test_peer_challenge_detects_independence():
    """Test that peer challenge detects independent reasoning."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"peer_challenge": {"enabled": True}})
    assessment = engine.peer_challenge(
        reviewer_id="model_2",
        target_id="model_1",
        task_id="task_1",
        target_contribution="The answer is A because of unique mathematical derivation using calculus.",
        reviewer_contribution="The answer is B because of empirical experimental evidence from physics.",
    )
    assert assessment.independent_reasoning is True


def test_collective_reflection_basic():
    """Test basic collective reflection."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"collective_reflection": {"enabled": True}})
    reflection = engine.collective_reflect(
        task_id="task_1",
        participating_models=["model_1", "model_2", "model_3"],
        contributions={
            "model_1": "I believe the answer is A.",
            "model_2": "I believe the answer is B.",
            "model_3": "I believe the answer is A.",
        },
        final_answer="A",
        verification_results=[],
    )
    assert reflection.task_id == "task_1"
    assert len(reflection.participating_models) == 3


def test_collective_reflection_detects_groupthink():
    """Test that collective reflection detects groupthink."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"collective_reflection": {"enabled": True}})
    reflection = engine.collective_reflect(
        task_id="task_1",
        participating_models=["model_1", "model_2", "model_3"],
        contributions={
            "model_1": "The answer is A.",
            "model_2": "The answer is A.",
            "model_3": "The answer is A.",
        },
        final_answer="A",
        verification_results=[],
    )
    assert reflection.premature_convergence is True
    assert "correlated" in reflection.potential_correlated_failure.lower()


def test_collective_reflection_identifies_missing_info():
    """Test that collective reflection identifies missing information."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"collective_reflection": {"enabled": True}})
    reflection = engine.collective_reflect(
        task_id="task_1",
        participating_models=["model_1"],
        contributions={"model_1": "I think the answer is A."},
        final_answer="A",
        verification_results=[],
    )
    assert len(reflection.missing_information) > 0


def test_belief_revision():
    """Test belief revision recording."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"belief_revision": {"enabled": True}})
    revision = engine.record_belief_revision(
        model_id="model_1",
        task_id="task_1",
        original_belief="The answer is A.",
        revised_belief="The answer is B.",
        evidence_that_changed_mind="Model 2 provided counterevidence.",
        other_model_influenced_by="model_2",
    )
    assert revision.model_id == "model_1"
    assert revision.revised_belief == "The answer is B."
    assert revision.revision_justified is True


def test_collective_confidence():
    """Test collective confidence computation."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"collective_reflection": {"enabled": True}})
    reflection = engine.collective_reflect(
        task_id="task_1",
        participating_models=["model_1", "model_2"],
        contributions={
            "model_1": "I believe the answer is A with high confidence.",
            "model_2": "I believe the answer is A with high confidence.",
        },
        final_answer="A",
        verification_results=[],
    )
    assert 0.0 <= reflection.collective_confidence <= 1.0


def test_self_reflection_report_generation():
    """Test that self-reflection generates structured reports."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"self_reflection": {"enabled": True}})
    assessment = engine.self_assess(
        model_id="model_1",
        task_id="task_1",
        contribution="I believe the answer is A.",
        evidence=["evidence X"],
        confidence=0.7,
    )
    d = assessment.to_dict()
    assert "model_id" in d
    assert "confidence" in d
    assert "strengths" in d
    assert "weaknesses" in d


def test_collective_reflection_most_useful_model():
    """Test identification of most useful model."""
    from src.training.self_reflection import SelfReflectionEngine

    engine = SelfReflectionEngine({"collective_reflection": {"enabled": True}})
    reflection = engine.collective_reflect(
        task_id="task_1",
        participating_models=["model_1", "model_2"],
        contributions={
            "model_1": "Short.",
            "model_2": "Longer detailed contribution with more reasoning.",
        },
        final_answer="A",
        verification_results=[],
    )
    assert reflection.most_useful_model == "model_2"
