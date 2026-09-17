"""Tests for information gain estimation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_information_gain_estimator_novel_capability():
    """Test that novel capabilities score high on information gain."""
    from src.training.information_gain import InformationGainEstimator

    estimator = InformationGainEstimator({
        "information_gain": {
            "novelty_weight": 0.3,
            "reliability_weight": 0.2,
            "disagreement_weight": 0.3,
            "verification_weight": 0.2,
        }
    })
    population = {
        "model_1": [0.9, 0.9, 0.9, 0.9],
        "model_2": [0.9, 0.9, 0.9, 0.9],
    }
    gain = estimator.estimate_gain(
        model_id="model_3",
        capability_vector={"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1},
        reliability=0.9,
        population_capabilities=population,
        verification_accuracy=0.8,
    )
    assert gain.composite_gain > 0.0


def test_information_gain_estimator_redundant_capability():
    """Test that redundant capabilities score low on information gain."""
    from src.training.information_gain import InformationGainEstimator

    estimator = InformationGainEstimator({
        "information_gain": {
            "novelty_weight": 0.3,
            "reliability_weight": 0.2,
            "disagreement_weight": 0.3,
            "verification_weight": 0.2,
        }
    })
    population = {
        "model_1": [0.9, 0.9, 0.9, 0.9],
    }
    gain = estimator.estimate_gain(
        model_id="model_2",
        capability_vector={"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9},
        reliability=0.9,
        population_capabilities=population,
        verification_accuracy=0.8,
    )
    assert gain.capability_novelty < 0.5


def test_information_gain_ranking():
    """Test ranking by information gain."""
    from src.training.information_gain import InformationGainEstimator

    estimator = InformationGainEstimator({
        "information_gain": {
            "novelty_weight": 0.3,
            "reliability_weight": 0.2,
            "disagreement_weight": 0.3,
            "verification_weight": 0.2,
        }
    })
    population = {
        "model_1": [0.9, 0.9, 0.9, 0.9],
    }
    candidates = [
        {"model_id": "a", "capability_vector": {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}, "reliability": 0.9, "verification_accuracy": 0.8},
        {"model_id": "b", "capability_vector": {"a": 0.1, "b": 0.1, "c": 0.1, "d": 0.1}, "reliability": 0.9, "verification_accuracy": 0.8},
    ]
    ranked = estimator.rank_by_information_gain(candidates, population)
    assert len(ranked) == 2
    assert ranked[0].composite_gain >= ranked[1].composite_gain
