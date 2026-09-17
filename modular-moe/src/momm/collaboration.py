"""Collaborative synthesis for Prothynesis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

from .communication import PeerMessage, PeerCommunicationProtocol, MessageType
from .memory import MemoryManager, MemoryRecord


class UncertaintyLevel(Enum):
    CERTAIN = "certain"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    NOT_VERIFIED = "not_verified"


class SynthesisStrategy(Enum):
    EVIDENCE_WEIGHTED = "evidence_weighted"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    PEER_CONSENSUS = "peer_consensus"
    HIERARCHICAL = "hierarchical"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"


@dataclass
class CandidateResult:
    """A candidate result from a model."""
    candidate_id: str
    model_id: str
    task_id: str
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    critiques: List[str] = field(default_factory=list)
    verifications: List[str] = field(default_factory=list)
    revisions: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    assumptions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    uncertainty: str = UncertaintyLevel.NOT_VERIFIED.value
    verification_status: str = "pending"
    verification_plan: List[str] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    memory_references: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SynthesisResult:
    """Result of collaborative synthesis."""
    task_id: str
    final_candidate: str
    confidence: float
    participating_models: List[str]
    total_candidates: int
    synthesis_strategy: str
    evidence_quality: float
    consensus_score: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


class CollaborativeSynthesizer:
    """
    Collaborative synthesis engine.
    
    Combines multiple model candidates into a final result
    using evidence-based consensus, not majority voting.
    """
    
    def __init__(
        self,
        task_id: str,
        strategy: str = SynthesisStrategy.EVIDENCE_WEIGHTED.value,
    ):
        self.task_id = task_id
        self.strategy = SynthesisStrategy(strategy)
        self._candidates: Dict[str, CandidateResult] = {}
        self._messages: List[PeerMessage] = []
    
    def add_candidate(
        self,
        model_id: str,
        content: str,
        confidence: float,
        evidence: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CandidateResult:
        """Add a candidate result from a model."""
        candidate_id = f"candidate_{len(self._candidates) + 1:03d}"
        
        candidate = CandidateResult(
            candidate_id=candidate_id,
            model_id=model_id,
            task_id=self.task_id,
            content=content,
            confidence=confidence,
            evidence=evidence or [],
            metadata=metadata or {},
        )
        
        self._candidates[candidate_id] = candidate
        return candidate
    
    def add_critique(
        self,
        candidate_id: str,
        critique: str,
        model_id: str,
        confidence: float = 0.8,
    ) -> None:
        """Add a critique to a candidate."""
        if candidate_id in self._candidates:
            self._candidates[candidate_id].critiques.append(critique)
    
    def add_verification(
        self,
        candidate_id: str,
        verified: bool,
        model_id: str,
        confidence: float = 0.9,
    ) -> None:
        """Add a verification to a candidate."""
        if candidate_id in self._candidates:
            self._candidates[candidate_id].verifications.append(
                f"{model_id}:{verified}:{confidence}"
            )
    
    def add_revision(
        self,
        candidate_id: str,
        revision: str,
        model_id: str,
    ) -> None:
        """Add a revision to a candidate."""
        if candidate_id in self._candidates:
            self._candidates[candidate_id].revisions.append(revision)
            # Update content to latest revision
            self._candidates[candidate_id].content = revision
    
    def record_peer_message(self, message: PeerMessage) -> None:
        """Record a peer message in the synthesis context."""
        self._messages.append(message)
    
    def synthesize(self) -> SynthesisResult:
        """
        Perform collaborative synthesis.
        
        Uses evidence-based consensus, not majority voting.
        """
        if not self._candidates:
            return SynthesisResult(
                task_id=self.task_id,
                final_candidate="",
                confidence=0.0,
                participating_models=[],
                total_candidates=0,
                synthesis_strategy=self.strategy.value,
                evidence_quality=0.0,
                consensus_score=0.0,
            )
        
        participating_models = list(set(c.model_id for c in self._candidates.values()))
        
        # Score each candidate
        scored_candidates = []
        for candidate in self._candidates.values():
            score = self._score_candidate(candidate)
            scored_candidates.append((score, candidate))
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select best candidate
        best_score, best_candidate = scored_candidates[0]
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus(scored_candidates)
        
        # Calculate evidence quality
        evidence_quality = self._calculate_evidence_quality()
        
        return SynthesisResult(
            task_id=self.task_id,
            final_candidate=best_candidate.content,
            confidence=best_candidate.confidence,
            participating_models=participating_models,
            total_candidates=len(self._candidates),
            synthesis_strategy=self.strategy.value,
            evidence_quality=evidence_quality,
            consensus_score=consensus_score,
            metadata={
                "best_candidate_id": best_candidate.candidate_id,
                "best_model_id": best_candidate.model_id,
                "best_score": best_score,
                "all_scores": {c.candidate_id: s for s, c in scored_candidates},
            }
        )
    
    def _score_candidate(self, candidate: CandidateResult) -> float:
        """Score a candidate based on multiple factors."""
        score = 0.0
        
        # Base confidence
        score += candidate.confidence * 0.3
        
        # Evidence quality
        evidence_score = min(1.0, len(candidate.evidence) * 0.1)
        score += evidence_score * 0.3
        
        # Verification status
        verified_count = sum(1 for v in candidate.verifications if ":True:" in v)
        verification_score = min(1.0, verified_count * 0.2)
        score += verification_score * 0.2
        
        # Revision count (fewer revisions = more stable)
        revision_penalty = min(0.1, len(candidate.revisions) * 0.02)
        score -= revision_penalty
        
        # Critique penalty
        critique_penalty = min(0.1, len(candidate.critiques) * 0.02)
        score -= critique_penalty
        
        return max(0.0, min(1.0, score))
    
    def _calculate_consensus(self, scored_candidates) -> float:
        """Calculate consensus score among candidates."""
        if len(scored_candidates) <= 1:
            return 1.0
        
        scores = [s for s, _ in scored_candidates]
        max_score = max(scores)
        min_score = min(scores)
        
        if max_score == min_score:
            return 1.0
        
        # Consensus is high when scores are close
        spread = max_score - min_score
        consensus = max(0.0, 1.0 - spread)
        
        return consensus
    
    def _calculate_evidence_quality(self) -> float:
        """Calculate overall evidence quality."""
        total_evidence = sum(len(c.evidence) for c in self._candidates.values())
        total_verifications = sum(len(c.verifications) for c in self._candidates.values())
        
        if not self._candidates:
            return 0.0
        
        avg_evidence = total_evidence / len(self._candidates)
        avg_verifications = total_verifications / len(self._candidates)
        
        quality = min(1.0, (avg_evidence * 0.1 + avg_verifications * 0.2))
        return quality
    
    def get_all_candidates(self) -> List[CandidateResult]:
        """Get all candidates."""
        return list(self._candidates.values())
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize synthesis state."""
        return {
            "task_id": self.task_id,
            "strategy": self.strategy.value,
            "candidates": {cid: c.to_dict() for cid, c in self._candidates.items()},
            "message_count": len(self._messages),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "CollaborativeSynthesizer":
        """Deserialize synthesis state."""
        synthesizer = cls(
            task_id=data["task_id"],
            strategy=data.get("strategy", SynthesisStrategy.EVIDENCE_WEIGHTED.value),
        )
        
        for cid, cdata in data.get("candidates", {}).items():
            candidate = CandidateResult(**cdata)
            synthesizer._candidates[cid] = candidate
        
        return synthesizer


@dataclass
class Contradiction:
    """Represents a contradiction between candidates."""
    contradiction_id: str
    task_id: str
    candidate_a_id: str
    candidate_b_id: str
    claim_a: str
    claim_b: str
    assumption_a: str = ""
    assumption_b: str = ""
    evidence_a: List[str] = field(default_factory=list)
    evidence_b: List[str] = field(default_factory=list)
    resolution: str = ""
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class ContradictionResolver:
    """
    Resolves contradictions between candidates.
    
    When candidates disagree, find why they differ,
    what assumption differs, what evidence differs,
    then request targeted verification.
    """

    def __init__(self, config: dict):
        self.config = config
        self.contradiction_threshold = config.get("contradiction_resolution", {}).get("contradiction_threshold", 0.3)
        self._contradictions: Dict[str, Contradiction] = {}

    def detect_contradictions(self, candidates: List[CandidateResult]) -> List[Contradiction]:
        """Detect contradictions between candidates."""
        contradictions = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                a = candidates[i]
                b = candidates[j]
                if a.content != b.content and a.task_id == b.task_id:
                    contradiction = Contradiction(
                        contradiction_id=f"contradiction_{a.candidate_id}_{b.candidate_id}_{int(time.time()*1000)}",
                        task_id=a.task_id,
                        candidate_a_id=a.candidate_id,
                        candidate_b_id=b.candidate_id,
                        claim_a=a.content,
                        claim_b=b.content,
                        assumption_a="; ".join(a.assumptions) if a.assumptions else "",
                        assumption_b="; ".join(b.assumptions) if b.assumptions else "",
                        evidence_a=a.evidence[:3],
                        evidence_b=b.evidence[:3],
                    )
                    contradictions.append(contradiction)
                    self._contradictions[contradiction.contradiction_id] = contradiction
        return contradictions

    def resolve_contradiction(
        self,
        contradiction: Contradiction,
        resolver_id: str,
        resolve_fn: Callable[[Contradiction, str], str],
    ) -> Contradiction:
        """Resolve a contradiction using a resolver."""
        resolution = resolve_fn(contradiction, resolver_id)
        contradiction.resolution = resolution
        contradiction.resolved = True
        return contradiction

    def get_unresolved_contradictions(self) -> List[Contradiction]:
        return [c for c in self._contradictions.values() if not c.resolved]

    def get_contradiction_rate(self, task_id: str) -> float:
        task_contradictions = [
            c for c in self._contradictions.values() if c.task_id == task_id
        ]
        if not task_contradictions:
            return 0.0
        return len(task_contradictions) / max(1, len(self._contradictions))
