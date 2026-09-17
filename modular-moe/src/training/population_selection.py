"""Population selection engine for MrTP candidate ranking and retention."""

from __future__ import annotations

import math
import hashlib
import itertools
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from enum import Enum

from .evolution import ModelFitness
from ..utils.paths import is_allowed_dataset_path

logger = logging.getLogger(__name__)


class CandidateStatus(Enum):
    RETAINED = "retained"
    DISCARDED = "discarded"
    ARCHIVED = "archived"
    PENDING = "pending"


class ModelFamily(Enum):
    """Specialized model families."""
    GENERAL_SOLVER = "general_solver"
    VERIFIER = "verifier"
    ADVERSARIAL = "adversarial"
    VISION = "vision"
    COMPUTER_USE = "computer_use"
    TOOL = "tool"


@dataclass
class CapabilityVector:
    general_knowledge: float = 0.0
    language: float = 0.0
    mathematics: float = 0.0
    programming: float = 0.0
    computer_science: float = 0.0
    science: float = 0.0
    reasoning: float = 0.0
    verification: float = 0.0
    planning: float = 0.0
    tool_use: float = 0.0
    memory: float = 0.0
    collaboration: float = 0.0
    computer_vision: float = 0.0
    video_understanding: float = 0.0
    computer_use: float = 0.0
    adversarial_reasoning: float = 0.0
    fact_checking: float = 0.0
    proof_checking: float = 0.0
    code_review: float = 0.0
    assumption_detection: float = 0.0
    contradiction_detection: float = 0.0
    uncertainty_estimation: float = 0.0
    self_correction: float = 0.0
    calibration: float = 0.0
    information_gain: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "CapabilityVector":
        return cls(**{k: float(v) for k, v in d.items() if k in cls.__dataclass_fields__})

    def cosine_similarity(self, other: "CapabilityVector") -> float:
        a = [getattr(self, f) for f in other.__dataclass_fields__]
        b = [getattr(other, f) for f in other.__dataclass_fields__]
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) + 1e-10
        nb = math.sqrt(sum(x * x for x in b)) + 1e-10
        return dot / (na * nb)


@dataclass
class CandidateRecord:
    model_id: str
    model_family: str = "general_solver"
    lineage: Dict[str, Any] = field(default_factory=dict)
    training_run: Dict[str, Any] = field(default_factory=dict)
    dataset_ownership: Dict[str, Any] = field(default_factory=dict)
    capability_vector: CapabilityVector = field(default_factory=CapabilityVector)
    overall_quality: float = 0.0
    domain_score: float = 0.0
    reasoning_score: float = 0.0
    verification_score: float = 0.0
    reliability: float = 0.0
    diversity: float = 0.0
    complementarity: float = 0.0
    redundancy_penalty: float = 0.0
    lineage_penalty: float = 0.0
    composite_score: float = 0.0
    status: CandidateStatus = CandidateStatus.PENDING
    rejection_reason: Optional[str] = None
    failure_categories: List[str] = field(default_factory=list)
    benchmark_results: Dict[str, Any] = field(default_factory=dict)
    parameter_count: int = 0
    checkpoint_path: str = ""
    created_at: float = field(default_factory=__import__("time").time)
    information_gain: float = 0.0
    contribution_score: float = 0.0
    calibration_error: float = 0.0
    uncertainty_level: str = "not_verified"
    verification_status: str = "pending"
    peer_endorsements: int = 0
    peer_rejections: int = 0
    tool_use_success_rate: float = 0.0
    hallucination_rate: float = 0.0
    error_rate: float = 0.0
    reputation_tier: str = "uninitialized"
    hard_examples_yielded: int = 0
    verifier_accuracy: float = 0.0
    adversarial_effectiveness: float = 0.0
    restart_count: int = 0

    def compute_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        if weights is None:
            weights = {
                "overall_quality": 0.20,
                "domain_score": 0.10,
                "reasoning_score": 0.12,
                "verification_score": 0.10,
                "reliability": 0.10,
                "diversity": 0.08,
                "complementarity": 0.08,
                "information_gain": 0.08,
                "contribution_score": 0.06,
                "redundancy_penalty": -0.04,
                "lineage_penalty": -0.04,
            }
        score = (
            weights.get("overall_quality", 0.0) * self.overall_quality
            + weights.get("domain_score", 0.0) * self.domain_score
            + weights.get("reasoning_score", 0.0) * self.reasoning_score
            + weights.get("verification_score", 0.0) * self.verification_score
            + weights.get("reliability", 0.0) * self.reliability
            + weights.get("diversity", 0.0) * self.diversity
            + weights.get("complementarity", 0.0) * self.complementarity
            + weights.get("information_gain", 0.0) * self.information_gain
            + weights.get("contribution_score", 0.0) * self.contribution_score
            + weights.get("redundancy_penalty", 0.0) * self.redundancy_penalty
            + weights.get("lineage_penalty", 0.0) * self.lineage_penalty
        )
        self.composite_score = max(0.0, min(1.0, score))
        return self.composite_score

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["capability_vector"] = self.capability_vector.to_dict()
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CandidateRecord":
        data = dict(d)
        data["capability_vector"] = CapabilityVector.from_dict(data.pop("capability_vector", {}))
        data["status"] = CandidateStatus(data.pop("status", "pending"))
        return cls(**data)


class PopulationSelectionEngine:
    """Selects the strongest, most diverse, complementary candidates."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redundancy_threshold = config.get("redundancy_threshold", 0.92)
        self.diversity_threshold = config.get("diversity_threshold", 0.80)
        self.retention_ratio = config.get("retention_ratio", 0.10)
        self.min_retained_per_domain = config.get("min_retained_per_domain", 5)
        self.max_retained_per_domain = config.get("max_retained_per_domain", 10)
        self.quality_floor = config.get("quality_floor", 0.50)
        self.capability_weights = config.get(
            "capability_weights",
            {
                "general_knowledge": 0.10,
                "language": 0.08,
                "mathematics": 0.10,
                "programming": 0.08,
                "computer_science": 0.06,
                "science": 0.06,
                "reasoning": 0.12,
                "verification": 0.10,
                "planning": 0.06,
                "tool_use": 0.04,
                "memory": 0.04,
                "collaboration": 0.04,
                "computer_vision": 0.04,
                "video_understanding": 0.02,
                "computer_use": 0.06,
            },
        )

    def score_candidate(self, candidate: CandidateRecord) -> CandidateRecord:
        """Compute composite score and normalized metadata."""
        cap = candidate.capability_vector
        domain_score = sum(
            getattr(cap, k, 0.0) * v for k, v in self.capability_weights.items()
        )
        candidate.domain_score = min(1.0, domain_score)
        candidate.reasoning_score = cap.reasoning * 0.5 + cap.planning * 0.3 + cap.verification * 0.2
        candidate.verification_score = cap.verification * 0.6 + cap.reasoning * 0.4
        candidate.reliability = min(1.0, candidate.overall_quality * 0.7 + candidate.verification_score * 0.3)

        if candidate.model_family == ModelFamily.VERIFIER.value:
            candidate.verification_score = min(1.0, candidate.verification_score * 1.2)
            candidate.reliability = min(1.0, candidate.reliability * 1.1)

        if candidate.model_family == ModelFamily.ADVERSARIAL.value:
            candidate.adversarial_effectiveness = cap.adversarial_reasoning * 0.5 + cap.assumption_detection * 0.3 + cap.contradiction_detection * 0.2
            candidate.contribution_score = max(candidate.contribution_score, candidate.adversarial_effectiveness)

        candidate.compute_composite()
        return candidate

    def compute_information_gain(self, candidate: CandidateRecord, population: List[CandidateRecord]) -> None:
        if not population:
            candidate.information_gain = 1.0
            return
        my_vec = list(candidate.capability_vector.to_dict().values())
        population_vecs = [list(c.capability_vector.to_dict().values()) for c in population if c.model_id != candidate.model_id]
        if not population_vecs:
            candidate.information_gain = 1.0
            return
        import math
        min_dist = min(
            math.sqrt(sum((a - b) ** 2 for a, b in zip(my_vec, pv)))
            for pv in population_vecs
        )
        max_dist = math.sqrt(len(my_vec))
        candidate.information_gain = max(0.0, min(1.0, min_dist / max(max_dist, 1e-10)))

    def compute_diversity_scores(self, candidates: List[CandidateRecord]) -> None:
        """Assign diversity score based on pairwise capability similarity."""
        if len(candidates) < 2:
            for c in candidates:
                c.diversity = 1.0
            return
        for c in candidates:
            similarities = []
            for other in candidates:
                if c.model_id == other.model_id:
                    continue
                sim = c.capability_vector.cosine_similarity(other.capability_vector)
                similarities.append(sim)
            avg_sim = sum(similarities) / len(similarities)
            c.diversity = max(0.0, min(1.0, 1.0 - avg_sim))

    def compute_complementarity(self, candidate: CandidateRecord, population: List[CandidateRecord]) -> None:
        """Complementarity is higher when candidate covers capabilities weak in population."""
        if not population:
            candidate.complementarity = 1.0
            return
        pop_caps = [c.capability_vector for c in population if c.model_id != candidate.model_id]
        if not pop_caps:
            candidate.complementarity = 1.0
            return
        avg_pop = CapabilityVector(
            **{
                f: sum(getattr(cv, f, 0.0) for cv in pop_caps) / len(pop_caps)
                for f in CapabilityVector.__dataclass_fields__
            }
        )
        weak_caps = [f for f in CapabilityVector.__dataclass_fields__ if avg_pop.__dict__[f] < 0.3]
        if not weak_caps:
            candidate.complementarity = 0.5
            return
        coverage_gain = sum(getattr(candidate.capability_vector, f, 0.0) for f in weak_caps)
        candidate.complementarity = min(1.0, coverage_gain / len(weak_caps))

    def compute_redundancy_penalty(self, candidate: CandidateRecord, population: List[CandidateRecord]) -> None:
        max_sim = 0.0
        for other in population:
            if candidate.model_id == other.model_id:
                continue
            if other.model_family != candidate.model_family:
                continue
            sim = candidate.capability_vector.cosine_similarity(other.capability_vector)
            max_sim = max(max_sim, sim)
        redundant = max_sim > self.redundancy_threshold
        candidate.redundancy_penalty = max_sim if redundant else 0.0

    def compute_lineage_penalty(self, candidate: CandidateRecord, population: List[CandidateRecord]) -> None:
        lineage_key = candidate.lineage.get("initialization_source", "")
        if lineage_key in ("fresh", ""):
            candidate.lineage_penalty = 0.0
            return
        shared = sum(1 for c in population if c.lineage.get("initialization_source") == lineage_key)
        candidate.lineage_penalty = min(0.5, shared / max(1, len(population)) * 0.5)

    def compute_contribution_score(self, candidate: CandidateRecord) -> None:
        score = 0.0
        score += candidate.peer_endorsements * 0.05
        score -= candidate.peer_rejections * 0.05
        score += candidate.hard_examples_yielded * 0.1
        score += candidate.verifier_accuracy * 0.3
        score += candidate.tool_use_success_rate * 0.2
        score += max(0.0, 1.0 - candidate.hallucination_rate) * 0.2
        score += max(0.0, 1.0 - candidate.error_rate) * 0.2
        candidate.contribution_score = max(0.0, min(1.0, score))

    def select(
        self,
        candidates: List[CandidateRecord],
        target_count: Optional[int] = None,
        model_family: Optional[str] = None,
    ) -> Tuple[List[CandidateRecord], List[CandidateRecord]]:
        """Rank and select candidates. Returns (retained, discarded)."""
        if not candidates:
            return [], []

        pool = [c for c in candidates if model_family is None or c.model_family == model_family]
        scored = [self.score_candidate(c) for c in pool]
        self.compute_diversity_scores(scored)
        for c in scored:
            self.compute_complementarity(c, scored)
            self.compute_redundancy_penalty(c, scored)
            self.compute_lineage_penalty(c, scored)
            self.compute_information_gain(c, scored)
            self.compute_contribution_score(c)
            c.compute_composite()

        qualified = [c for c in scored if c.overall_quality >= self.quality_floor]
        rejected = [c for c in scored if c.overall_quality < self.quality_floor]
        for c in rejected:
            c.status = CandidateStatus.DISCARDED
            c.rejection_reason = f"below_quality_floor({self.quality_floor})"

        qualified.sort(key=lambda c: c.composite_score, reverse=True)

        domain_groups: Dict[str, List[CandidateRecord]] = {}
        for c in qualified:
            domain = c.model_family
            domain_groups.setdefault(domain, []).append(c)

        retained: List[CandidateRecord] = []
        discarded: List[CandidateRecord] = []

        for domain, group in domain_groups.items():
            group.sort(key=lambda c: c.composite_score, reverse=True)
            max_keep = target_count or max(
                self.min_retained_per_domain,
                min(self.max_retained_per_domain, int(len(group) * self.retention_ratio)),
            )
            if len(group) <= max_keep:
                retained.extend(group)
                continue

            selected: List[CandidateRecord] = []
            for c in group:
                if len(selected) >= max_keep:
                    c.status = CandidateStatus.DISCARDED
                    c.rejection_reason = "exceeded_retention_limit"
                    discarded.append(c)
                    continue
                if not selected:
                    selected.append(c)
                    continue
                max_sim = max(
                    c.capability_vector.cosine_similarity(s.capability_vector) for s in selected
                )
                if max_sim < self.redundancy_threshold:
                    selected.append(c)
                else:
                    c.status = CandidateStatus.DISCARDED
                    c.rejection_reason = f"redundant(similarity={max_sim:.3f})"
                    discarded.append(c)
            retained.extend(selected)

        for c in retained:
            c.status = CandidateStatus.RETAINED
        return retained, discarded

    def build_hierarchy(
        self,
        retained: List[CandidateRecord],
        hierarchy_spec: Dict[str, int],
    ) -> Dict[str, List[CandidateRecord]]:
        """Assign retained candidates to hierarchy tiers."""
        tiers = {k: [] for k in hierarchy_spec}
        tier_names = ["ultimate", "master", "chief", "orchestral"]
        counts = [hierarchy_spec.get(t, 0) for t in tier_names]
        total = sum(counts)
        if total == 0 or not retained:
            return tiers

        # Sort by composite score descending
        sorted_cands = sorted(retained, key=lambda c: c.composite_score, reverse=True)

        idx = 0
        for tier, count in zip(tier_names, counts):
            for _ in range(count):
                if idx < len(sorted_cands):
                    tiers[tier].append(sorted_cands[idx])
                    idx += 1
        return tiers

    def export_selection_report(self, retained: List[CandidateRecord], discarded: List[CandidateRecord],
                                output_path: str) -> Dict[str, Any]:
        out = Path(output_path)
        if not is_allowed_dataset_path(out):
            raise PermissionError(f"Refusing to write outside approved roots: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "retained_count": len(retained),
            "discarded_count": len(discarded),
            "retained": [c.to_dict() for c in retained],
            "discarded": [c.to_dict() for c in discarded],
        }
        with open(out, "w") as f:
            import json
            json.dump(report, f, indent=2)
        logger.info(f"Selection report written: {out}")
        return report
