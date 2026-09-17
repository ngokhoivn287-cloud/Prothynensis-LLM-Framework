"""Recursive verification and meta-verification for Prothynesis.

Implements:
- RecursiveVerification: multi-round verification with revision
- MetaVerifier: higher-level independent assessment
- VerificationResult: structured verification outcome
- VerificationLadder: escalating verification depth
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class VerificationOutcome(Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"
    NEEDS_REVISION = "needs_revision"
    NEEDS_RECHECK = "needs_recheck"


class VerificationLevel(Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"
    EXHAUSTIVE = "exhaustive"


@dataclass
class VerificationResult:
    """Structured result of a verification round."""
    verification_id: str
    candidate_id: str
    verifier_id: str
    outcome: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions_checked: List[str] = field(default_factory=list)
    contradictions_found: List[str] = field(default_factory=list)
    tool_verifications: List[Dict[str, Any]] = field(default_factory=list)
    revision_suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecursiveVerificationState:
    """Tracks state through recursive verification rounds."""
    candidate_id: str
    original_content: str
    current_content: str
    round: int = 0
    max_rounds: int = 3
    outcomes: List[VerificationResult] = field(default_factory=list)
    revisions: List[str] = field(default_factory=list)
    converged: bool = False
    final_outcome: str = VerificationOutcome.UNCERTAIN.value

    def to_dict(self) -> dict:
        return asdict(self)


class RecursiveVerifier:
    """
    Performs recursive verification with revision.
    
    Flow:
    1. Verify candidate
    2. If FAIL/NEEDS_REVISION: request revision
    3. Re-verify revised candidate
    4. Repeat until convergence or max rounds
    """

    def __init__(self, config: dict):
        self.config = config
        self.max_rounds = config.get("recursive_verification", {}).get("max_rounds", 3)
        self.convergence_threshold = config.get("recursive_verification", {}).get("convergence_threshold", 0.9)
        self.uncertainty_threshold = config.get("recursive_verification", {}).get("uncertainty_threshold", 0.3)

    def verify(
        self,
        candidate_id: str,
        content: str,
        verifier_id: str,
        verification_fn: Callable[[str, str], VerificationResult],
    ) -> RecursiveVerificationState:
        """
        Perform recursive verification.
        
        Args:
            candidate_id: Candidate identifier
            content: Candidate content to verify
            verifier_id: Verifier model identifier
            verification_fn: Callable(content, verifier_id) -> VerificationResult
        """
        state = RecursiveVerificationState(
            candidate_id=candidate_id,
            original_content=content,
            current_content=content,
            max_rounds=self.max_rounds,
        )

        for round_num in range(self.max_rounds):
            state.round = round_num + 1
            result = verification_fn(state.current_content, verifier_id)
            state.outcomes.append(result)

            if result.outcome == VerificationOutcome.PASS.value:
                if result.confidence >= self.convergence_threshold:
                    state.converged = True
                    state.final_outcome = VerificationOutcome.PASS.value
                    return state

            if result.outcome == VerificationOutcome.FAIL.value:
                if not result.revision_suggestions:
                    state.final_outcome = VerificationOutcome.FAIL.value
                    return state
                revision = "; ".join(result.revision_suggestions)
                state.revisions.append(revision)
                state.current_content = self._apply_revision(state.current_content, revision)
                continue

            if result.outcome == VerificationOutcome.NEEDS_REVISION.value:
                revision = "; ".join(result.revision_suggestions) if result.revision_suggestions else "revise based on feedback"
                state.revisions.append(revision)
                state.current_content = self._apply_revision(state.current_content, revision)
                continue

            if result.outcome == VerificationOutcome.NEEDS_RECHECK.value:
                continue

            state.final_outcome = result.outcome
            return state

        state.final_outcome = VerificationOutcome.UNCERTAIN.value
        return state

    def _apply_revision(self, content: str, revision: str) -> str:
        """Apply revision to content. Placeholder for actual revision logic."""
        return f"{content}\n[REVISION {revision}]"

    def get_verification_trace(self, state: RecursiveVerificationState) -> Dict[str, Any]:
        return {
            "candidate_id": state.candidate_id,
            "rounds": state.round,
            "converged": state.converged,
            "final_outcome": state.final_outcome,
            "revision_count": len(state.revisions),
            "outcomes": [o.to_dict() for o in state.outcomes],
        }


class MetaVerifier:
    """
    Higher-level verification stage.
    
    Independently assesses evidence and consistency
    without simply repeating the candidate.
    """

    def __init__(self, config: dict):
        self.config = config
        self.min_evidence_quality = config.get("meta_verifier", {}).get("min_evidence_quality", 0.5)
        self.consistency_threshold = config.get("meta_verifier", {}).get("consistency_threshold", 0.8)

    def meta_verify(
        self,
        candidate_id: str,
        content: str,
        evidence: List[str],
        verifications: List[VerificationResult],
        verifier_id: str,
        meta_verification_fn: Callable[[str, List[str], List[VerificationResult]], VerificationResult],
    ) -> VerificationResult:
        """
        Perform meta-verification.
        
        Args:
            candidate_id: Candidate identifier
            content: Final candidate content
            evidence: List of evidence strings
            verifications: List of prior verification results
            verifier_id: Meta-verifier identifier
            meta_verification_fn: Callable(content, evidence, verifications) -> VerificationResult
        """
        evidence_quality = self._estimate_evidence_quality(evidence)
        consistency = self._estimate_consistency(verifications)

        if evidence_quality < self.min_evidence_quality:
            return VerificationResult(
                verification_id=f"meta_{candidate_id}_{int(time.time()*1000)}",
                candidate_id=candidate_id,
                verifier_id=verifier_id,
                outcome=VerificationOutcome.UNCERTAIN.value,
                confidence=evidence_quality,
                evidence=evidence,
                metadata={"reason": "insufficient_evidence_quality", "evidence_quality": evidence_quality},
            )

        if consistency < self.consistency_threshold:
            return VerificationResult(
                verification_id=f"meta_{candidate_id}_{int(time.time()*1000)}",
                candidate_id=candidate_id,
                verifier_id=verifier_id,
                outcome=VerificationOutcome.NEEDS_RECHECK.value,
                confidence=consistency,
                evidence=evidence,
                metadata={"reason": "inconsistent_verifications", "consistency": consistency},
            )

        result = meta_verification_fn(content, evidence, verifications)
        result.verification_id = f"meta_{candidate_id}_{int(time.time()*1000)}"
        result.metadata.setdefault("evidence_quality", evidence_quality)
        result.metadata.setdefault("consistency", consistency)
        return result

    def _estimate_evidence_quality(self, evidence: List[str]) -> float:
        if not evidence:
            return 0.0
        return min(1.0, len(evidence) / 10.0)

    def _estimate_consistency(self, verifications: List[VerificationResult]) -> float:
        if not verifications:
            return 0.0
        outcomes = [v.outcome for v in verifications]
        if not outcomes:
            return 0.0
        most_common = max(set(outcomes), key=outcomes.count)
        return outcomes.count(most_common) / len(outcomes)
