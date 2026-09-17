"""Information gain estimation for adaptive recruitment.

Estimates expected information gain from recruiting additional models
to guide dynamic recruitment decisions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class ModelInfoGain:
    """Estimated information gain from a model."""
    model_id: str
    expected_information_gain: float
    capability_novelty: float
    reliability: float
    disagreement_potential: float
    verification_value: float
    composite_gain: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class InformationGainEstimator:
    """
    Estimates expected information gain from recruiting models.
    
    Prefers models likely to add new information over redundant ones.
    """

    def __init__(self, config: dict):
        self.config = config
        self.novelty_weight = config.get("information_gain", {}).get("novelty_weight", 0.3)
        self.reliability_weight = config.get("information_gain", {}).get("reliability_weight", 0.2)
        self.disagreement_weight = config.get("information_gain", {}).get("disagreement_weight", 0.3)
        self.verification_weight = config.get("information_gain", {}).get("verification_weight", 0.2)

    def estimate_gain(
        self,
        model_id: str,
        capability_vector: Dict[str, float],
        reliability: float,
        population_capabilities: Dict[str, List[float]],
        verification_accuracy: float = 0.5,
    ) -> ModelInfoGain:
        """
        Estimate expected information gain from a model.
        
        Args:
            model_id: Model identifier
            capability_vector: Model capability scores
            reliability: Model reliability [0, 1]
            population_capabilities: Dict of model_id -> capability vectors for population
            verification_accuracy: Model's verification accuracy
        """
        capability_novelty = self._compute_capability_novelty(
            capability_vector, population_capabilities
        )
        disagreement_potential = self._compute_disagreement_potential(
            capability_vector, population_capabilities
        )

        composite = (
            self.novelty_weight * capability_novelty
            + self.reliability_weight * reliability
            + self.disagreement_weight * disagreement_potential
            + self.verification_weight * verification_accuracy
        )

        return ModelInfoGain(
            model_id=model_id,
            expected_information_gain=composite,
            capability_novelty=capability_novelty,
            reliability=reliability,
            disagreement_potential=disagreement_potential,
            verification_value=verification_accuracy,
            composite_gain=composite,
        )

    def rank_by_information_gain(
        self,
        candidates: List[Dict[str, Any]],
        population_capabilities: Dict[str, List[float]],
    ) -> List[ModelInfoGain]:
        gains = []
        for candidate in candidates:
            gain = self.estimate_gain(
                model_id=candidate["model_id"],
                capability_vector=candidate.get("capability_vector", {}),
                reliability=candidate.get("reliability", 0.5),
                population_capabilities=population_capabilities,
                verification_accuracy=candidate.get("verification_accuracy", 0.5),
            )
            gains.append(gain)
        gains.sort(key=lambda g: g.composite_gain, reverse=True)
        return gains

    def _compute_capability_novelty(
        self,
        capability_vector: Dict[str, float],
        population_capabilities: Dict[str, List[float]],
    ) -> float:
        if not population_capabilities:
            return 1.0
        if not capability_vector:
            return 0.0
        population_vecs = list(population_capabilities.values())
        if not population_vecs:
            return 1.0
        my_vec = list(capability_vector.values())
        min_distances = []
        for pop_vec in population_vecs:
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(my_vec, pop_vec)))
            min_distances.append(dist)
        max_distance = math.sqrt(len(my_vec))
        novelty = min(min_distances) / max(max_distance, 1e-10)
        return max(0.0, min(1.0, novelty))

    def _compute_disagreement_potential(
        self,
        capability_vector: Dict[str, float],
        population_capabilities: Dict[str, List[float]],
    ) -> float:
        if not population_capabilities or not capability_vector:
            return 0.0
        my_vec = list(capability_vector.values())
        if not my_vec:
            return 0.0
        disagreements = []
        for pop_vec in population_capabilities.values():
            if not pop_vec or len(pop_vec) != len(my_vec):
                continue
            agreement = sum(1 for a, b in zip(my_vec, pop_vec) if abs(a - b) < 0.1)
            disagreement = 1.0 - (agreement / len(my_vec))
            disagreements.append(disagreement)
        if not disagreements:
            return 0.0
        return sum(disagreements) / len(disagreements)
