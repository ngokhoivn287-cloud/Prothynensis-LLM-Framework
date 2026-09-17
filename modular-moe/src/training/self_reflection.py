"""Self-reflection and collective deliberation for Prothynensis.

Implements structured self-assessment, peer challenge, belief revision,
collective confidence, and skeptical review.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class ConfidenceLevel(Enum):
    CERTAIN = "certain"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    NOT_VERIFIED = "not_verified"


@dataclass
class SelfAssessment:
    """A model's self-assessment of its own contribution."""
    model_id: str
    task_id: str
    confidence: float
    reasoning_quality: float
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    evidence_quality: float = 0.0
    potential_mistakes: List[str] = field(default_factory=list)
    what_would_change_mind: List[str] = field(default_factory=list)
    contributed_value: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PeerAssessment:
    """A model's assessment of another model."""
    reviewer_id: str
    target_id: str
    task_id: str
    strongest_argument: str = ""
    caught_missed_error: bool = False
    confident_without_evidence: bool = False
    independent_reasoning: bool = False
    meaningful_disagreement: bool = False
    trust_for_task: float = 0.0
    should_not_trust: bool = False
    trust_reason: str = ""
    helped_most: bool = False
    hurt_group: bool = False
    hurt_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollectiveReflection:
    """Collective self-reflection of the group."""
    task_id: str
    participating_models: List[str] = field(default_factory=list)
    why_we_agreed: str = ""
    shared_assumptions: List[str] = field(default_factory=list)
    strongest_counterargument: str = ""
    missing_information: List[str] = field(default_factory=list)
    potential_correlated_failure: str = ""
    collaboration_improved_result: bool = False
    premature_convergence: bool = False
    recommended_verification: List[str] = field(default_factory=list)
    recommended_additional_model: str = ""
    recommended_additional_capability: str = ""
    expected_benefit_of_additional_model: str = ""
    should_stop: bool = False
    stop_reason: str = ""
    collective_confidence: float = 0.0
    confidence_justification: str = ""
    what_could_make_us_wrong: List[str] = field(default_factory=list)
    process_weaknesses: List[str] = field(default_factory=list)
    architecture_weaknesses: List[str] = field(default_factory=list)
    recommended_changes: List[str] = field(default_factory=list)
    strongest_evidence: List[str] = field(default_factory=list)
    weakest_assumption: str = ""
    main_disagreement: str = ""
    most_useful_model: str = ""
    most_useful_contribution: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BeliefRevision:
    """Record of a belief revision event."""
    model_id: str
    task_id: str
    original_belief: str
    revised_belief: str
    evidence_that_changed_mind: str = ""
    other_model_influenced_by: str = ""
    revision_justified: bool = False
    too_resistant: bool = False
    too_easily_influenced: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class SelfReflectionEngine:
    """
    Engine for self-reflection and collective deliberation.
    
    Implements:
    - Self-assessment
    - Peer challenge
    - Collective reflection
    - Belief revision
    - Collective confidence
    - Skeptical review
    """

    def __init__(self, config: dict):
        self.config = config
        self._self_reflection_enabled = config.get("self_reflection", {}).get("enabled", True)
        self._peer_challenge_enabled = config.get("peer_challenge", {}).get("enabled", True)
        self._collective_reflection_enabled = config.get("collective_reflection", {}).get("enabled", True)
        self._belief_revision_enabled = config.get("belief_revision", {}).get("enabled", True)
        self._skeptical_review_enabled = config.get("skeptical_review", {}).get("enabled", True)
        self._self_assessments: Dict[str, SelfAssessment] = {}
        self._peer_assessments: Dict[str, PeerAssessment] = {}
        self._collective_reflections: Dict[str, CollectiveReflection] = {}
        self._belief_revisions: Dict[str, BeliefRevision] = {}

    def self_assess(
        self,
        model_id: str,
        task_id: str,
        contribution: str,
        evidence: List[str],
        confidence: float,
    ) -> SelfAssessment:
        """Perform self-assessment for a model."""
        assessment = SelfAssessment(
            model_id=model_id,
            task_id=task_id,
            confidence=confidence,
            reasoning_quality=min(1.0, confidence * 0.9),
            strengths=self._infer_strengths(contribution, evidence),
            weaknesses=self._infer_weaknesses(contribution, confidence),
            assumptions=self._extract_assumptions(contribution),
            uncertainties=self._extract_uncertainties(contribution),
            evidence_quality=len(evidence) / max(1, len(evidence) + 1),
            potential_mistakes=self._infer_potential_mistakes(contribution, confidence),
            what_would_change_mind=self._infer_change_mind_triggers(contribution),
            contributed_value=contribution[:500],
        )
        self._self_assessments[f"{model_id}_{task_id}"] = assessment
        return assessment

    def peer_challenge(
        self,
        reviewer_id: str,
        target_id: str,
        task_id: str,
        target_contribution: str,
        reviewer_contribution: str,
    ) -> PeerAssessment:
        """Perform peer challenge assessment."""
        assessment = PeerAssessment(
            reviewer_id=reviewer_id,
            target_id=target_id,
            task_id=task_id,
            strongest_argument=target_contribution[:500],
            caught_missed_error=self._detect_caught_error(target_contribution, reviewer_contribution),
            confident_without_evidence=self._detect_overconfidence(target_contribution),
            independent_reasoning=self._detect_independence(target_contribution, reviewer_contribution),
            meaningful_disagreement=self._detect_meaningful_disagreement(target_contribution, reviewer_contribution),
            trust_for_task=self._compute_trust(target_contribution),
            should_not_trust=self._detect_misleading(target_contribution),
            trust_reason=self._generate_trust_reason(target_contribution),
        )
        key = f"{reviewer_id}_{target_id}_{task_id}"
        self._peer_assessments[key] = assessment
        return assessment

    def collective_reflect(
        self,
        task_id: str,
        participating_models: List[str],
        contributions: Dict[str, str],
        final_answer: str,
        verification_results: List[Any],
    ) -> CollectiveReflection:
        """Perform collective self-reflection."""
        reflection = CollectiveReflection(
            task_id=task_id,
            participating_models=participating_models,
            why_we_agreed=self._analyze_agreement_reason(contributions),
            shared_assumptions=self._extract_shared_assumptions(contributions),
            strongest_counterargument=self._find_strongest_counterargument(contributions, final_answer),
            missing_information=self._identify_missing_information(contributions),
            potential_correlated_failure=self._detect_correlated_failure(contributions),
            collaboration_improved_result=self._assess_collaboration_gain(contributions, final_answer),
            premature_convergence=self._detect_premature_convergence(contributions),
            recommended_verification=self._recommend_verification(contributions, verification_results),
            recommended_additional_model=self._recommend_additional_model(contributions),
            recommended_additional_capability=self._recommend_additional_capability(contributions),
            expected_benefit_of_additional_model=self._estimate_benefit_of_additional_model(contributions),
            should_stop=self._should_stop(contributions, verification_results),
            stop_reason=self._stop_reason(contributions, verification_results),
            collective_confidence=self._compute_collective_confidence(contributions, verification_results),
            confidence_justification=self._justify_confidence(contributions, verification_results),
            what_could_make_us_wrong=self._identify_failure_modes(contributions),
            process_weaknesses=self._identify_process_weaknesses(contributions),
            architecture_weaknesses=self._identify_architecture_weaknesses(contributions),
            recommended_changes=self._recommend_changes(contributions, verification_results),
            strongest_evidence=self._extract_strongest_evidence(contributions),
            weakest_assumption=self._identify_weakest_assumption(contributions),
            main_disagreement=self._identify_main_disagreement(contributions),
            most_useful_model=self._identify_most_useful_model(contributions),
            most_useful_contribution=self._identify_most_useful_contribution(contributions),
        )
        self._collective_reflections[task_id] = reflection
        return reflection

    def record_belief_revision(
        self,
        model_id: str,
        task_id: str,
        original_belief: str,
        revised_belief: str,
        evidence_that_changed_mind: str = "",
        other_model_influenced_by: str = "",
    ) -> BeliefRevision:
        """Record a belief revision event."""
        revision = BeliefRevision(
            model_id=model_id,
            task_id=task_id,
            original_belief=original_belief,
            revised_belief=revised_belief,
            evidence_that_changed_mind=evidence_that_changed_mind,
            other_model_influenced_by=other_model_influenced_by,
            revision_justified=bool(evidence_that_changed_mind),
            too_resistant=not bool(evidence_that_changed_mind) and original_belief != revised_belief,
            too_easily_influenced=bool(evidence_that_changed_mind) and original_belief == revised_belief,
        )
        key = f"{model_id}_{task_id}_{int(time.time()*1000)}"
        self._belief_revisions[key] = revision
        return revision

    def get_self_assessment(self, model_id: str, task_id: str) -> Optional[SelfAssessment]:
        return self._self_assessments.get(f"{model_id}_{task_id}")

    def get_peer_assessment(self, reviewer_id: str, target_id: str, task_id: str) -> Optional[PeerAssessment]:
        return self._peer_assessments.get(f"{reviewer_id}_{target_id}_{task_id}")

    def get_collective_reflection(self, task_id: str) -> Optional[CollectiveReflection]:
        return self._collective_reflections.get(task_id)

    def get_belief_revisions(self, task_id: str) -> List[BeliefRevision]:
        return [r for r in self._belief_revisions.values() if r.task_id == task_id]

    def _infer_strengths(self, contribution: str, evidence: List[str]) -> List[str]:
        strengths = []
        if len(evidence) > 2:
            strengths.append("evidence-based reasoning")
        if "verify" in contribution.lower() or "check" in contribution.lower():
            strengths.append("verification awareness")
        if "assume" in contribution.lower():
            strengths.append("assumption awareness")
        if "uncertain" in contribution.lower() or "unknown" in contribution.lower():
            strengths.append("uncertainty acknowledgment")
        if not strengths:
            strengths.append("contributed to discussion")
        return strengths

    def _infer_weaknesses(self, contribution: str, confidence: float) -> List[str]:
        weaknesses = []
        if confidence > 0.9 and len(contribution) < 100:
            weaknesses.append("high confidence with limited elaboration")
        if "error" not in contribution.lower() and "mistake" not in contribution.lower():
            weaknesses.append("did not explicitly consider own errors")
        if "assume" not in contribution.lower():
            weaknesses.append("did not state assumptions explicitly")
        return weaknesses

    def _extract_assumptions(self, contribution: str) -> List[str]:
        assumptions = []
        for keyword in ["assume", "assumption", "presume", "suppose", "if"]:
            if keyword in contribution.lower():
                assumptions.append(f"uses '{keyword}' language")
        return assumptions or ["assumptions not explicitly stated"]

    def _extract_uncertainties(self, contribution: str) -> List[str]:
        uncertainties = []
        for keyword in ["uncertain", "unknown", "unclear", "not sure", "might"]:
            if keyword in contribution.lower():
                uncertainties.append(keyword)
        return uncertainties or ["uncertainty not explicitly stated"]

    def _infer_potential_mistakes(self, contribution: str, confidence: float) -> List[str]:
        mistakes = []
        if confidence > 0.8:
            mistakes.append("overconfidence risk")
        if "verify" not in contribution.lower():
            mistakes.append("insufficient verification")
        if "evidence" not in contribution.lower():
            mistakes.append("evidence gap")
        return mistakes or ["standard reasoning risks"]

    def _infer_change_mind_triggers(self, contribution: str) -> List[str]:
        triggers = []
        if "evidence" in contribution.lower():
            triggers.append("contradictory evidence")
        if "assume" in contribution.lower():
            triggers.append("assumption failure")
        if "verify" in contribution.lower():
            triggers.append("verification failure")
        return triggers or ["contradictory evidence"]

    def _detect_caught_error(self, target: str, reviewer: str) -> bool:
        target_lower = target.lower()
        reviewer_lower = reviewer.lower()
        error_terms = ["error", "mistake", "wrong", "incorrect", "invalid", "flaw"]
        for term in error_terms:
            if term in reviewer_lower and term not in target_lower:
                return True
        return False

    def _detect_overconfidence(self, contribution: str) -> bool:
        confidence_terms = ["certainly", "definitely", "always", "never", "obviously", "clearly"]
        contribution_lower = contribution.lower()
        confidence_count = sum(1 for term in confidence_terms if term in contribution_lower)
        return confidence_count >= 2

    def _detect_independence(self, target: str, reviewer: str) -> bool:
        target_words = set(target.lower().split())
        reviewer_words = set(reviewer.lower().split())
        if not target_words or not reviewer_words:
            return False
        overlap = len(target_words & reviewer_words)
        total = len(target_words | reviewer_words)
        return (overlap / total) < 0.3

    def _detect_meaningful_disagreement(self, target: str, reviewer: str) -> bool:
        return target.lower().strip() != reviewer.lower().strip() and len(target) > 10 and len(reviewer) > 10

    def _compute_trust(self, contribution: str) -> float:
        trust = 0.5
        if "evidence" in contribution.lower():
            trust += 0.1
        if "assume" in contribution.lower():
            trust += 0.1
        if "uncertain" in contribution.lower():
            trust += 0.1
        if "verify" in contribution.lower():
            trust += 0.1
        if self._detect_overconfidence(contribution):
            trust -= 0.2
        return max(0.0, min(1.0, trust))

    def _detect_misleading(self, contribution: str) -> bool:
        misleading_terms = ["always", "never", "every", "none", "all", "no one"]
        return any(term in contribution.lower() for term in misleading_terms)

    def _generate_trust_reason(self, contribution: str) -> str:
        if self._detect_overconfidence(contribution):
            return "overconfident language without sufficient evidence"
        if "evidence" in contribution.lower():
            return "cites evidence and reasoning"
        return "standard contribution"

    def _analyze_agreement_reason(self, contributions: Dict[str, str]) -> str:
        values = list(contributions.values())
        if len(values) < 2:
            return "single contribution"
        unique = len(set(values))
        if unique == 1:
            return "all contributions identical - possible groupthink or shared assumptions"
        return f"{unique} distinct contributions out of {len(values)}"

    def _extract_shared_assumptions(self, contributions: Dict[str, str]) -> List[str]:
        all_text = " ".join(contributions.values()).lower()
        assumptions = []
        if "assume" in all_text:
            assumptions.append("models use assumption-based reasoning")
        if "evidence" in all_text:
            assumptions.append("evidence is considered important")
        if "verify" in all_text:
            assumptions.append("verification is valued")
        return assumptions or ["no explicit shared assumptions detected"]

    def _find_strongest_counterargument(self, contributions: Dict[str, str], final_answer: str) -> str:
        for contrib in contributions.values():
            if contrib.lower() != final_answer.lower() and len(contrib) > 20:
                return contrib[:500]
        return "no explicit counterargument found"

    def _identify_missing_information(self, contributions: Dict[str, str]) -> List[str]:
        missing = []
        all_text = " ".join(contributions.values()).lower()
        if "data" not in all_text and "evidence" not in all_text:
            missing.append("empirical data")
        if "verify" not in all_text:
            missing.append("independent verification")
        if "assume" not in all_text:
            missing.append("explicit assumption checking")
        return missing or ["no critical missing information identified"]

    def _detect_correlated_failure(self, contributions: Dict[str, str]) -> str:
        values = list(contributions.values())
        if len(values) < 2:
            return "insufficient data"
        unique = len(set(values))
        if unique == 1:
            return "all models produced identical outputs - high correlated failure risk"
        return "no obvious correlated failure detected"

    def _assess_collaboration_gain(self, contributions: Dict[str, str], final_answer: str) -> bool:
        if len(contributions) <= 1:
            return False
        unique = len(set(contributions.values()))
        return unique > 1

    def _detect_premature_convergence(self, contributions: Dict[str, str]) -> bool:
        values = list(contributions.values())
        if len(values) < 2:
            return False
        unique = len(set(values))
        return unique == 1

    def _recommend_verification(self, contributions: Dict[str, str], verification_results: List[Any]) -> List[str]:
        recommendations = []
        if not verification_results:
            recommendations.append("run verification")
        for contrib in contributions.values():
            if "assume" in contrib.lower() and "verify assumption" not in recommendations:
                recommendations.append("verify key assumptions")
        return recommendations or ["standard verification"]

    def _recommend_additional_model(self, contributions: Dict[str, str]) -> str:
        all_text = " ".join(contributions.values()).lower()
        if "math" in all_text or "calculate" in all_text:
            return "mathematical verifier"
        if "code" in all_text or "program" in all_text:
            return "code reviewer"
        if "evidence" in all_text:
            return "fact checker"
        return "independent verifier"

    def _recommend_additional_capability(self, contributions: Dict[str, str]) -> str:
        all_text = " ".join(contributions.values()).lower()
        if "evidence" in all_text:
            return "fact_checking"
        if "verify" in all_text:
            return "verification"
        if "assume" in all_text:
            return "assumption_detection"
        return "general reasoning"

    def _estimate_benefit_of_additional_model(self, contributions: Dict[str, str]) -> str:
        if len(set(contributions.values())) == 1:
            return "high - current consensus may be correlated"
        return "moderate - additional perspective could resolve remaining uncertainty"

    def _should_stop(self, contributions: Dict[str, str], verification_results: List[Any]) -> bool:
        if not verification_results:
            return False
        passed = sum(1 for v in verification_results if getattr(v, "outcome", "") == "pass")
        return passed >= len(verification_results) * 0.8

    def _stop_reason(self, contributions: Dict[str, str], verification_results: List[Any]) -> str:
        if not verification_results:
            return "insufficient verification"
        passed = sum(1 for v in verification_results if getattr(v, "outcome", "") == "pass")
        if passed >= len(verification_results) * 0.8:
            return "verification sufficient"
        return "verification insufficient"

    def _compute_collective_confidence(self, contributions: Dict[str, str], verification_results: List[Any]) -> float:
        if not contributions:
            return 0.0
        base_confidence = sum(len(c) for c in contributions.values()) / max(1, len(contributions) * 100)
        if verification_results:
            passed = sum(1 for v in verification_results if getattr(v, "outcome", "") == "pass")
            base_confidence *= (passed / max(1, len(verification_results)))
        return max(0.0, min(1.0, base_confidence))

    def _justify_confidence(self, contributions: Dict[str, str], verification_results: List[Any]) -> str:
        if not contributions:
            return "no contributions"
        if verification_results:
            passed = sum(1 for v in verification_results if getattr(v, "outcome", "") == "pass")
            return f"{passed}/{len(verification_results)} verifications passed"
        return f"{len(contributions)} contributions with no verification"

    def _identify_failure_modes(self, contributions: Dict[str, str]) -> List[str]:
        failure_modes = []
        values = list(contributions.values())
        if len(values) >= 2 and len(set(values)) == 1:
            failure_modes.append("groupthink/correlated assumptions")
        if any("confident" in c.lower() for c in values):
            failure_modes.append("overconfidence")
        if any("assume" in c.lower() for c in values):
            failure_modes.append("assumption reliance")
        return failure_modes or ["standard failure modes"]

    def _identify_process_weaknesses(self, contributions: Dict[str, str]) -> List[str]:
        weaknesses = []
        if len(contributions) < 3:
            weaknesses.append("insufficient model diversity")
        values = list(contributions.values())
        if len(values) >= 2 and len(set(values)) == 1:
            weaknesses.append("premature consensus")
        return weaknesses or ["no obvious process weaknesses identified"]

    def _identify_architecture_weaknesses(self, contributions: Dict[str, str]) -> List[str]:
        weaknesses = []
        all_text = " ".join(contributions.values()).lower()
        if "verify" not in all_text:
            weaknesses.append("verification may be underweighted")
        if "memory" not in all_text:
            weaknesses.append("memory/reuse not leveraged")
        if "tool" not in all_text:
            weaknesses.append("tool use not utilized")
        return weaknesses or ["no critical architecture weaknesses identified"]

    def _recommend_changes(self, contributions: Dict[str, str], verification_results: List[Any]) -> List[str]:
        changes = []
        if not verification_results:
            changes.append("add verification step")
        if len(set(contributions.values())) == 1:
            changes.append("increase model diversity")
        if any("assume" in c.lower() for c in contributions.values()):
            changes.append("explicitly test assumptions")
        return changes or ["continue current approach"]

    def _extract_strongest_evidence(self, contributions: Dict[str, str]) -> List[str]:
        evidence = []
        for contrib in contributions.values():
            if "evidence" in contrib.lower():
                evidence.append(contrib[:200])
        return evidence or ["no explicit evidence stated"]

    def _identify_weakest_assumption(self, contributions: Dict[str, str]) -> str:
        for contrib in contributions.values():
            if "assume" in contrib.lower():
                return contrib[:200]
        return "no explicit assumptions identified"

    def _identify_main_disagreement(self, contributions: Dict[str, str]) -> str:
        values = list(contributions.values())
        if len(values) < 2:
            return "no disagreement"
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i].lower() != values[j].lower():
                    return f"'{values[i][:100]}' vs '{values[j][:100]}'"
        return "no significant disagreement"

    def _identify_most_useful_model(self, contributions: Dict[str, str]) -> str:
        if not contributions:
            return "none"
        best = max(contributions.items(), key=lambda x: len(x[1]))
        return best[0]

    def _identify_most_useful_contribution(self, contributions: Dict[str, str]) -> str:
        if not contributions:
            return "none"
        best = max(contributions.items(), key=lambda x: len(x[1]))
        return best[1][:200]
