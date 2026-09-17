"""Collective Reasoning Workspace (CRW) core module for Prothynesis.

Implements a three-layer workspace for multi-model collective reasoning:
- L0: raw reasoning event stream
- L1: collective working state
- L2: long-term memory references

Supports typed events, context slots, snapshot/restore, and token-budgeted
context construction. Integrates with the Prothynensis reasoning stack.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

if TYPE_CHECKING:
    from ..training.calibration import CalibrationTracker, UncertaintyEstimator
    from ..training.correlated_failures import CorrelatedFailureDetector
    from ..training.information_gain import InformationGainEstimator
    from ..training.reputation import ReputationStore
    from ..training.self_reflection import SelfReflectionEngine
    from ..training.verification import MetaVerifier, RecursiveVerifier
    from ..momm.collaboration import CollaborativeSynthesizer
    from ..momm.communication import PeerCommunicationProtocol
    from ..momm.memory import MemoryManager
    from ..momm.recruitment import DynamicRecruiter, RecruitmentContext
    from ..momm.task import TaskManager


# ---------------------------------------------------------------------------
# Approved persistence roots (paths.py policy)
# ---------------------------------------------------------------------------
try:
    from ..utils.paths import get_temp_root, is_allowed_dataset_path

    _PATHS_AVAILABLE = True
except ImportError:
    _PATHS_AVAILABLE = False
    get_temp_root = None  # type: ignore[assignment]
    is_allowed_dataset_path = None  # type: ignore[assignment]


def _get_workspace_root() -> Path:
    """Return the approved workspace persistence root."""
    if _PATHS_AVAILABLE and get_temp_root is not None:
        root = get_temp_root() / "collective_workspace"
        root.mkdir(parents=True, exist_ok=True)
        return root
    # Fallback to a local directory outside the package
    fallback = Path(__file__).resolve().parent.parent / "workspace_state"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


# ---------------------------------------------------------------------------
# Event taxonomy
# ---------------------------------------------------------------------------
class CollectiveEventType(Enum):
    """Typed events in the collective reasoning stream."""

    PROPOSAL = "proposal"
    ARGUMENT = "argument"
    COUNTERARGUMENT = "counterargument"
    QUESTION = "question"
    EVIDENCE = "evidence"
    CRITIQUE = "critique"
    VERIFICATION = "verification"
    REVISION = "revision"
    AGREEMENT = "agreement"
    DISAGREEMENT = "disagreement"
    WARNING = "warning"
    RECRUITMENT_REQUEST = "recruitment_request"
    STOP_REQUEST = "stop_request"
    SUMMARY = "summary"


# ---------------------------------------------------------------------------
# L0: Raw reasoning stream
# ---------------------------------------------------------------------------
@dataclass
class CollectiveEvent:
    """A single event in the L0 raw reasoning stream."""

    event_id: str
    event_type: str
    task_id: str
    session_id: str
    sender_id: str
    target_ids: List[str]
    content: str
    confidence: float = 0.0
    evidence_refs: List[str] = field(default_factory=list)
    memory_refs: List[str] = field(default_factory=list)
    parent_event_id: Optional[str] = None
    layer: str = "L0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CollectiveEvent":
        return cls(**d)


# ---------------------------------------------------------------------------
# L1: Collective working state
# ---------------------------------------------------------------------------
@dataclass
class WorkingHypothesis:
    """A hypothesis maintained in the collective working state."""

    hypothesis_id: str
    content: str
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    supporters: List[str] = field(default_factory=list)
    challengers: List[str] = field(default_factory=list)
    revision_count: int = 0
    status: str = "active"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkingHypothesis":
        return cls(**d)


@dataclass
class CollectiveWorkingState:
    """L1 collective working state for a task session."""

    task_id: str
    session_id: str
    active_hypotheses: Dict[str, WorkingHypothesis] = field(default_factory=dict)
    consensus_candidate: Optional[str] = None
    consensus_confidence: float = 0.0
    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    recruitment_requests: List[Dict[str, Any]] = field(default_factory=list)
    stop_requests: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["active_hypotheses"] = {k: v.to_dict() for k, v in self.active_hypotheses.items()}
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "CollectiveWorkingState":
        hypotheses = {k: WorkingHypothesis.from_dict(v) for k, v in d.pop("active_hypotheses", {}).items()}
        obj = cls(**d)
        obj.active_hypotheses = hypotheses
        return obj


# ---------------------------------------------------------------------------
# L2: Long-term memory references
# ---------------------------------------------------------------------------
@dataclass
class MemoryReference:
    """Reference to a long-term memory entry in L2."""

    ref_id: str
    memory_id: str
    memory_type: str
    task_id: str
    session_id: str
    relevance_score: float = 0.0
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryReference":
        return cls(**d)


@dataclass
class LongTermMemoryContext:
    """L2 long-term memory references for a task session."""

    task_id: str
    session_id: str
    references: Dict[str, MemoryReference] = field(default_factory=dict)
    total_references: int = 0
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["references"] = {k: v.to_dict() for k, v in self.references.items()}
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "LongTermMemoryContext":
        refs = {k: MemoryReference.from_dict(v) for k, v in d.pop("references", {}).items()}
        obj = cls(**d)
        obj.references = refs
        return obj


# ---------------------------------------------------------------------------
# Context slots
# ---------------------------------------------------------------------------
@dataclass
class ContextSlot:
    """A context slot acquired by a model for reasoning."""

    slot_id: str
    model_id: str
    task_id: str
    session_id: str
    acquired_at: str
    released_at: Optional[str] = None
    token_budget: int = 4096
    tokens_used: int = 0
    event_ids: List[str] = field(default_factory=list)
    handoff_to: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_active(self) -> bool:
        """Return True if the slot is currently held."""
        return self.released_at is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContextSlot":
        return cls(**d)


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------
@dataclass
class WorkspaceSnapshot:
    """Point-in-time snapshot of the full workspace state."""

    snapshot_id: str
    task_id: str
    session_id: str
    layer0_events: List[Dict[str, Any]] = field(default_factory=list)
    layer1_state: Dict[str, Any] = field(default_factory=dict)
    layer2_refs: Dict[str, Any] = field(default_factory=dict)
    slots: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    token_budget_config: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "WorkspaceSnapshot":
        return cls(**d)


# ---------------------------------------------------------------------------
# Main workspace class
# ---------------------------------------------------------------------------
class CollectiveReasoningWorkspace:
    """Thread-safe collective reasoning workspace with three layers.

    Layers
    ------
    L0: Raw chronological stream of CollectiveEvent instances.
    L1: CollectiveWorkingState summarising the current deliberation.
    L2: LongTermMemoryContext with references to persistent memory.

    Context slots allow models to acquire bounded context windows with
    acquire / release / handoff semantics.  Token-budgeted context
    construction limits the number of events and references included in
    any assembled prompt.  Snapshot / restore allows the full workspace
    state to be captured and reinstantiated.

    Integration points (all optional):
        memory_manager      - MemoryManager for L2 reference creation
        task_manager        - TaskManager for task state tracking
        communication       - PeerCommunicationProtocol for peer messages
        synthesizer         - CollaborativeSynthesizer for candidate fusion
        recruiter           - DynamicRecruiter for adaptive recruitment
        reputation_store    - ReputationStore for model reliability
        info_gain_estimator - InformationGainEstimator
        calibration_tracker - CalibrationTracker
        uncertainty_estimator - UncertaintyEstimator
        recursive_verifier  - RecursiveVerifier
        meta_verifier       - MetaVerifier
        self_reflection     - SelfReflectionEngine
        failure_detector    - CorrelatedFailureDetector
    """

    def __init__(
        self,
        task_id: str,
        session_id: str,
        *,
        default_token_budget: int = 4096,
        max_events: int = 10000,
        max_slots: int = 32,
        max_l2_refs: int = 5000,
        persistence_enabled: bool = True,
        # Integrated components (optional)
        memory_manager: Optional["MemoryManager"] = None,
        task_manager: Optional["TaskManager"] = None,
        communication: Optional["PeerCommunicationProtocol"] = None,
        synthesizer: Optional["CollaborativeSynthesizer"] = None,
        recruiter: Optional["DynamicRecruiter"] = None,
        recruitment_context: Optional["RecruitmentContext"] = None,
        reputation_store: Optional["ReputationStore"] = None,
        info_gain_estimator: Optional["InformationGainEstimator"] = None,
        calibration_tracker: Optional["CalibrationTracker"] = None,
        uncertainty_estimator: Optional["UncertaintyEstimator"] = None,
        recursive_verifier: Optional["RecursiveVerifier"] = None,
        meta_verifier: Optional["MetaVerifier"] = None,
        self_reflection: Optional["SelfReflectionEngine"] = None,
        failure_detector: Optional["CorrelatedFailureDetector"] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.task_id = task_id
        self.session_id = session_id
        self.default_token_budget = default_token_budget
        self.max_events = max_events
        self.max_slots = max_slots
        self.max_l2_refs = max_l2_refs
        self.persistence_enabled = persistence_enabled
        self.config = config or {}

        # Thread safety
        self._lock = threading.RLock()

        # L0: raw event stream
        self._events: List[CollectiveEvent] = []

        # L1: collective working state
        self._l1 = CollectiveWorkingState(task_id=task_id, session_id=session_id)

        # L2: long-term memory references
        self._l2 = LongTermMemoryContext(task_id=task_id, session_id=session_id)

        # Context slots
        self._slots: Dict[str, ContextSlot] = {}
        self._model_slot_index: Dict[str, str] = {}

        # Snapshots
        self._snapshots: Dict[str, WorkspaceSnapshot] = {}

        # Integrated components (stored for lazy use)
        self._memory_manager = memory_manager
        self._task_manager = task_manager
        self._communication = communication
        self._synthesizer = synthesizer
        self._recruiter = recruiter
        self._recruitment_context = recruitment_context
        self._reputation_store = reputation_store
        self._info_gain_estimator = info_gain_estimator
        self._calibration_tracker = calibration_tracker
        self._uncertainty_estimator = uncertainty_estimator
        self._recursive_verifier = recursive_verifier
        self._meta_verifier = meta_verifier
        self._self_reflection = self_reflection
        self._failure_detector = failure_detector

        # Persistence path
        if persistence_enabled:
            self._state_path = _get_workspace_root() / f"crw_{task_id}_{session_id}.json"
        else:
            self._state_path = None

        # Load persisted state if present
        if self._state_path and self._state_path.exists():
            self._load_state()

    # ------------------------------------------------------------------
    # Event emission (L0 + L1 update)
    # ------------------------------------------------------------------
    def emit(
        self,
        event_type: Union[CollectiveEventType, str],
        sender_id: str,
        content: str,
        *,
        target_ids: Optional[List[str]] = None,
        confidence: float = 0.0,
        evidence_refs: Optional[List[str]] = None,
        memory_refs: Optional[List[str]] = None,
        parent_event_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CollectiveEvent:
        """Emit a typed event into L0 and update L1 accordingly.

        Thread-safe.
        """
        with self._lock:
            if isinstance(event_type, CollectiveEventType):
                event_type_str = event_type.value
            else:
                event_type_str = event_type

            event = CollectiveEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type_str,
                task_id=self.task_id,
                session_id=self.session_id,
                sender_id=sender_id,
                target_ids=target_ids or [],
                content=content,
                confidence=confidence,
                evidence_refs=evidence_refs or [],
                memory_refs=memory_refs or [],
                parent_event_id=parent_event_id,
                metadata=metadata or {},
            )

            self._events.append(event)
            self._prune_events()
            self._update_l1(event)
            self._persist_if_enabled()
            return event

    def _prune_events(self) -> None:
        """Enforce max_events limit on L0 stream."""
        while len(self._events) > self.max_events:
            self._events.pop(0)

    def _update_l1(self, event: CollectiveEvent) -> None:
        """Update L1 working state based on an incoming event."""
        self._l1.updated_at = datetime.now().isoformat()

        et = event.event_type
        if et == CollectiveEventType.PROPOSAL.value:
            hyp = WorkingHypothesis(
                hypothesis_id=str(uuid.uuid4()),
                content=event.content,
                confidence=event.confidence,
                supporters=[event.sender_id],
                metadata={"proposal_event_id": event.event_id},
            )
            self._l1.active_hypotheses[hyp.hypothesis_id] = hyp

        elif et == CollectiveEventType.ARGUMENT.value:
            self._attach_support(event)

        elif et == CollectiveEventType.COUNTERARGUMENT.value:
            self._attach_challenge(event)

        elif et == CollectiveEventType.EVIDENCE.value:
            self._attach_evidence(event)

        elif et == CollectiveEventType.CRITIQUE.value:
            self._attach_critique(event)

        elif et == CollectiveEventType.REVISION.value:
            self._apply_revision(event)

        elif et == CollectiveEventType.VERIFICATION.value:
            self._apply_verification(event)

        elif et == CollectiveEventType.AGREEMENT.value:
            self._record_agreement(event)

        elif et == CollectiveEventType.DISAGREEMENT.value:
            self._record_disagreement(event)

        elif et == CollectiveEventType.WARNING.value:
            self._l1.warnings.append(
                {
                    "event_id": event.event_id,
                    "sender_id": event.sender_id,
                    "content": event.content,
                    "timestamp": event.timestamp,
                    "metadata": event.metadata,
                }
            )

        elif et == CollectiveEventType.QUESTION.value:
            self._l1.open_questions.append(event.content)

        elif et == CollectiveEventType.RECRUITMENT_REQUEST.value:
            self._l1.recruitment_requests.append(
                {
                    "event_id": event.event_id,
                    "sender_id": event.sender_id,
                    "content": event.content,
                    "confidence": event.confidence,
                    "timestamp": event.timestamp,
                    "metadata": event.metadata,
                }
            )

        elif et == CollectiveEventType.STOP_REQUEST.value:
            self._l1.stop_requests.append(event.sender_id)

        elif et == CollectiveEventType.SUMMARY.value:
            self._l1.summary = event.content

    def _attach_support(self, event: CollectiveEvent) -> None:
        """Attach an argument as support to the most relevant hypothesis."""
        target = event.metadata.get("hypothesis_id")
        if target and target in self._l1.active_hypotheses:
            hyp = self._l1.active_hypotheses[target]
            hyp.supporters.append(event.sender_id)
            hyp.evidence.extend(event.evidence_refs)
            return
        # Fallback: attach to hypothesis with highest confidence
        if self._l1.active_hypotheses:
            best = max(self._l1.active_hypotheses.values(), key=lambda h: h.confidence)
            best.supporters.append(event.sender_id)
            best.evidence.extend(event.evidence_refs)

    def _attach_challenge(self, event: CollectiveEvent) -> None:
        """Attach a counterargument as a challenge."""
        target = event.metadata.get("hypothesis_id")
        if target and target in self._l1.active_hypotheses:
            hyp = self._l1.active_hypotheses[target]
            hyp.challengers.append(event.sender_id)

    def _attach_evidence(self, event: CollectiveEvent) -> None:
        """Attach evidence to a hypothesis or working state."""
        target = event.metadata.get("hypothesis_id")
        if target and target in self._l1.active_hypotheses:
            self._l1.active_hypotheses[target].evidence.extend(event.evidence_refs)
        else:
            if self._l1.active_hypotheses:
                best = max(self._l1.active_hypotheses.values(), key=lambda h: h.confidence)
                best.evidence.extend(event.evidence_refs)

    def _attach_critique(self, event: CollectiveEvent) -> None:
        """Record a critique in working state."""
        self._l1.disagreements.append(
            {
                "event_id": event.event_id,
                "sender_id": event.sender_id,
                "target_id": event.target_ids[0] if event.target_ids else None,
                "content": event.content,
                "confidence": event.confidence,
                "timestamp": event.timestamp,
                "metadata": event.metadata,
            }
        )

    def _apply_revision(self, event: CollectiveEvent) -> None:
        """Apply a revision to a hypothesis."""
        target = event.metadata.get("hypothesis_id")
        if target and target in self._l1.active_hypotheses:
            hyp = self._l1.active_hypotheses[target]
            hyp.content = event.content
            hyp.confidence = event.confidence
            hyp.revision_count += 1
            hyp.updated_at = datetime.now().isoformat()

    def _apply_verification(self, event: CollectiveEvent) -> None:
        """Apply verification outcome to a hypothesis."""
        target = event.metadata.get("hypothesis_id")
        if target and target in self._l1.active_hypotheses:
            hyp = self._l1.active_hypotheses[target]
            verified = event.metadata.get("verified", False)
            if verified:
                hyp.confidence = max(hyp.confidence, event.confidence)
            else:
                hyp.confidence = min(hyp.confidence, event.confidence * 0.5)

    def _record_agreement(self, event: CollectiveEvent) -> None:
        """Record agreement and potentially advance consensus."""
        target = event.metadata.get("hypothesis_id")
        if target and target in self._l1.active_hypotheses:
            hyp = self._l1.active_hypotheses[target]
            hyp.supporters.append(event.sender_id)
            self._recompute_consensus()

    def _record_disagreement(self, event: CollectiveEvent) -> None:
        """Record explicit disagreement."""
        self._l1.disagreements.append(
            {
                "event_id": event.event_id,
                "sender_id": event.sender_id,
                "target_id": event.target_ids[0] if event.target_ids else None,
                "content": event.content,
                "confidence": event.confidence,
                "timestamp": event.timestamp,
                "metadata": event.metadata,
            }
        )

    def _recompute_consensus(self) -> None:
        """Recompute consensus candidate from active hypotheses."""
        active = [h for h in self._l1.active_hypotheses.values() if h.status == "active"]
        if not active:
            return
        best = max(active, key=lambda h: h.confidence)
        self._l1.consensus_candidate = best.hypothesis_id
        self._l1.consensus_confidence = best.confidence

    # ------------------------------------------------------------------
    # L0 access
    # ------------------------------------------------------------------
    def get_events(
        self,
        *,
        event_type: Optional[Union[CollectiveEventType, str]] = None,
        sender_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[CollectiveEvent]:
        """Retrieve events from L0 with optional filtering."""
        with self._lock:
            results = list(self._events)
            if event_type is not None:
                if isinstance(event_type, CollectiveEventType):
                    ev = event_type.value
                else:
                    ev = event_type
                results = [e for e in results if e.event_type == ev]
            if sender_id is not None:
                results = [e for e in results if e.sender_id == sender_id]
            return results[offset : offset + limit]

    def get_event_by_id(self, event_id: str) -> Optional[CollectiveEvent]:
        """Return a single event by ID."""
        with self._lock:
            for e in self._events:
                if e.event_id == event_id:
                    return e
            return None

    def get_l0_stream(self) -> List[Dict[str, Any]]:
        """Return full L0 stream as list of dicts."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    # ------------------------------------------------------------------
    # L1 access
    # ------------------------------------------------------------------
    def get_working_state(self) -> CollectiveWorkingState:
        """Return the current L1 working state (thread-safe copy)."""
        with self._lock:
            return CollectiveWorkingState.from_dict(self._l1.to_dict())

    def get_hypothesis(self, hypothesis_id: str) -> Optional[WorkingHypothesis]:
        """Return a hypothesis by ID."""
        with self._lock:
            return self._l1.active_hypotheses.get(hypothesis_id)

    def get_consensus(self) -> Optional[Dict[str, Any]]:
        """Return the current consensus candidate if any."""
        with self._lock:
            if self._l1.consensus_candidate is None:
                return None
            hyp = self._l1.active_hypotheses.get(self._l1.consensus_candidate)
            if hyp is None:
                return None
            return {
                "hypothesis_id": hyp.hypothesis_id,
                "content": hyp.content,
                "confidence": hyp.confidence,
                "supporters": hyp.supporters,
                "revision_count": hyp.revision_count,
            }

    def get_open_questions(self) -> List[str]:
        """Return unresolved questions."""
        with self._lock:
            return list(self._l1.open_questions)

    def get_warnings(self) -> List[Dict[str, Any]]:
        """Return active warnings."""
        with self._lock:
            return list(self._l1.warnings)

    def get_disagreements(self) -> List[Dict[str, Any]]:
        """Return recorded disagreements."""
        with self._lock:
            return list(self._l1.disagreements)

    def get_recruitment_requests(self) -> List[Dict[str, Any]]:
        """Return pending recruitment requests."""
        with self._lock:
            return list(self._l1.recruitment_requests)

    def get_stop_requests(self) -> List[str]:
        """Return models that requested a stop."""
        with self._lock:
            return list(self._l1.stop_requests)

    def get_summary(self) -> Optional[str]:
        """Return the current session summary."""
        with self._lock:
            return self._l1.summary

    # ------------------------------------------------------------------
    # L2 access
    # ------------------------------------------------------------------
    def integrate_memory(
        self,
        memory_id: str,
        memory_type: str,
        relevance_score: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryReference:
        """Create an L2 memory reference for the current task/session.

        Thread-safe.
        """
        with self._lock:
            ref = MemoryReference(
                ref_id=str(uuid.uuid4()),
                memory_id=memory_id,
                memory_type=memory_type,
                task_id=self.task_id,
                session_id=self.session_id,
                relevance_score=relevance_score,
                metadata=metadata or {},
            )
            self._l2.references[ref.ref_id] = ref
            self._l2.total_references = len(self._l2.references)
            self._l2.updated_at = datetime.now().isoformat()
            self._prune_l2()
            self._persist_if_enabled()
            return ref

    def _prune_l2(self) -> None:
        """Enforce max_l2_refs limit."""
        while len(self._l2.references) > self.max_l2_refs:
            oldest = min(self._l2.references.values(), key=lambda r: r.last_accessed)
            del self._l2.references[oldest.ref_id]
            self._l2.total_references = len(self._l2.references)

    def get_memory_references(
        self,
        memory_type: Optional[str] = None,
        min_relevance: float = 0.0,
        limit: int = 100,
    ) -> List[MemoryReference]:
        """Return L2 references, optionally filtered."""
        with self._lock:
            refs = list(self._l2.references.values())
            if memory_type is not None:
                refs = [r for r in refs if r.memory_type == memory_type]
            if min_relevance > 0.0:
                refs = [r for r in refs if r.relevance_score >= min_relevance]
            refs.sort(key=lambda r: (r.relevance_score, r.last_accessed), reverse=True)
            return refs[:limit]

    def get_l2_context(self) -> Dict[str, Any]:
        """Return L2 context as a serialisable dict."""
        with self._lock:
            return self._l2.to_dict()

    # ------------------------------------------------------------------
    # Context slots: acquire / release / handoff
    # ------------------------------------------------------------------
    def acquire_slot(
        self,
        model_id: str,
        token_budget: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextSlot:
        """Acquire a context slot for *model_id*.

        Raises RuntimeError if the model already holds a slot or the pool
        is at capacity.

        Thread-safe.
        """
        with self._lock:
            if model_id in self._model_slot_index:
                raise RuntimeError(f"Model {model_id} already holds an active slot")
            if len(self._slots) >= self.max_slots:
                raise RuntimeError("Context slot pool at capacity")

            slot = ContextSlot(
                slot_id=str(uuid.uuid4()),
                model_id=model_id,
                task_id=self.task_id,
                session_id=self.session_id,
                acquired_at=datetime.now().isoformat(),
                token_budget=token_budget or self.default_token_budget,
                metadata=metadata or {},
            )
            self._slots[slot.slot_id] = slot
            self._model_slot_index[model_id] = slot.slot_id
            self._persist_if_enabled()
            return slot

    def release_slot(self, model_id: str) -> Optional[ContextSlot]:
        """Release the slot held by *model_id*.

        Thread-safe.
        """
        with self._lock:
            slot_id = self._model_slot_index.pop(model_id, None)
            if slot_id is None:
                return None
            slot = self._slots.pop(slot_id, None)
            if slot is not None:
                slot.released_at = datetime.now().isoformat()
            self._persist_if_enabled()
            return slot

    def handoff_slot(
        self,
        from_model_id: str,
        to_model_id: str,
    ) -> Optional[ContextSlot]:
        """Hand off a slot from one model to another without releasing it.

        Thread-safe.
        """
        with self._lock:
            slot_id = self._model_slot_index.get(from_model_id)
            if slot_id is None:
                return None
            slot = self._slots.get(slot_id)
            if slot is None:
                return None
            if to_model_id in self._model_slot_index:
                raise RuntimeError(f"Target model {to_model_id} already holds a slot")

            slot.handoff_to = to_model_id
            self._model_slot_index.pop(from_model_id, None)
            self._model_slot_index[to_model_id] = slot_id
            self._persist_if_enabled()
            return slot

    def get_slot(self, model_id: str) -> Optional[ContextSlot]:
        """Return the active slot for *model_id*, if any."""
        with self._lock:
            slot_id = self._model_slot_index.get(model_id)
            if slot_id is None:
                return None
            return self._slots.get(slot_id)

    def get_active_slots(self) -> List[ContextSlot]:
        """Return all currently active slots."""
        with self._lock:
            return [s for s in self._slots.values() if s.is_active()]

    def record_slot_usage(self, model_id: str, tokens_used: int) -> None:
        """Record token usage for the slot held by *model_id*."""
        with self._lock:
            slot_id = self._model_slot_index.get(model_id)
            if slot_id is not None and slot_id in self._slots:
                self._slots[slot_id].tokens_used = max(self._slots[slot_id].tokens_used, tokens_used)

    # ------------------------------------------------------------------
    # Token-budgeted context construction
    # ------------------------------------------------------------------
    def build_context(
        self,
        for_model_id: str,
        max_tokens: Optional[int] = None,
        include_l0: bool = True,
        include_l2: bool = True,
        l0_limit: int = 50,
        l2_limit: int = 20,
    ) -> Dict[str, Any]:
        """Build a token-budgeted context payload for *for_model_id*.

        Selects recent L0 events and high-relevance L2 references, then
        estimates token usage and truncates to *max_tokens*.

        Thread-safe.
        """
        with self._lock:
            budget = max_tokens or self.default_token_budget
            parts: List[str] = []
            token_est = 0

            if include_l0:
                events = self._events[-l0_limit:]
                l0_block: List[str] = []
                for ev in events:
                    line = f"[{ev.event_type}] {ev.sender_id}: {ev.content}"
                    est = self._estimate_tokens(line)
                    if token_est + est > budget and l0_block:
                        break
                    l0_block.append(line)
                    token_est += est
                if l0_block:
                    parts.append("\n".join(l0_block))

            if include_l2:
                refs = sorted(
                    self._l2.references.values(),
                    key=lambda r: (r.relevance_score, r.last_accessed),
                    reverse=True,
                )[:l2_limit]
                l2_block: List[str] = []
                for ref in refs:
                    line = f"[MEMORY:{ref.memory_type}] {ref.memory_id} (rel={ref.relevance_score:.2f})"
                    est = self._estimate_tokens(line)
                    if token_est + est > budget and l2_block:
                        break
                    l2_block.append(line)
                    token_est += est
                if l2_block:
                    parts.append("\n".join(l2_block))

            slot = self._slots.get(self._model_slot_index.get(for_model_id, ""))
            slot_info: Dict[str, Any] = {}
            if slot is not None:
                slot_info = {
                    "slot_id": slot.slot_id,
                    "token_budget": slot.token_budget,
                    "tokens_used": slot.tokens_used,
                }

            return {
                "task_id": self.task_id,
                "session_id": self.session_id,
                "for_model_id": for_model_id,
                "estimated_tokens": token_est,
                "max_tokens": budget,
                "slot": slot_info,
                "context": "\n---\n".join(parts) if parts else "",
                "l0_event_count": len(self._events),
                "l2_ref_count": len(self._l2.references),
            }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimation without heavy tokenizer dependency."""
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Snapshot / restore
    # ------------------------------------------------------------------
    def snapshot(self, snapshot_id: Optional[str] = None) -> WorkspaceSnapshot:
        """Capture the full workspace state as a snapshot.

        Thread-safe.
        """
        with self._lock:
            snap = WorkspaceSnapshot(
                snapshot_id=snapshot_id or str(uuid.uuid4()),
                task_id=self.task_id,
                session_id=self.session_id,
                layer0_events=[e.to_dict() for e in self._events],
                layer1_state=self._l1.to_dict(),
                layer2_refs=self._l2.to_dict(),
                slots={sid: s.to_dict() for sid, s in self._slots.items()},
                token_budget_config={
                    "default_token_budget": self.default_token_budget,
                    "max_events": self.max_events,
                    "max_slots": self.max_slots,
                    "max_l2_refs": self.max_l2_refs,
                },
                metadata={"source": "collective_workspace"},
            )
            self._snapshots[snap.snapshot_id] = snap
            return snap

    def restore(self, snapshot_id: str) -> None:
        """Restore workspace to a previously captured snapshot.

        Thread-safe.
        """
        with self._lock:
            snap = self._snapshots.get(snapshot_id)
            if snap is None:
                raise KeyError(f"Snapshot {snapshot_id} not found")

            self._events = [CollectiveEvent.from_dict(e) for e in snap.layer0_events]
            self._l1 = CollectiveWorkingState.from_dict(snap.layer1_state)
            self._l2 = LongTermMemoryContext.from_dict(snap.layer2_refs)
            self._slots = {sid: ContextSlot.from_dict(s) for sid, s in snap.slots.items()}
            self._model_slot_index = {}
            for slot in self._slots.values():
                if slot.is_active():
                    self._model_slot_index[slot.model_id] = slot.slot_id

            cfg = snap.token_budget_config
            if cfg:
                self.default_token_budget = cfg.get("default_token_budget", self.default_token_budget)
                self.max_events = cfg.get("max_events", self.max_events)
                self.max_slots = cfg.get("max_slots", self.max_slots)
                self.max_l2_refs = cfg.get("max_l2_refs", self.max_l2_refs)

            self._persist_if_enabled()

    def get_snapshot(self, snapshot_id: str) -> Optional[WorkspaceSnapshot]:
        """Return a stored snapshot without restoring."""
        with self._lock:
            return self._snapshots.get(snapshot_id)

    def list_snapshots(self) -> List[str]:
        """List all stored snapshot IDs."""
        with self._lock:
            return list(self._snapshots.keys())

    # ------------------------------------------------------------------
    # Integration helpers (lazy / optional)
    # ------------------------------------------------------------------
    def _ensure_memory_manager(self) -> "MemoryManager":
        if self._memory_manager is None:
            raise RuntimeError("MemoryManager not configured")
        return self._memory_manager

    def integrate_with_memory(
        self,
        content: str,
        memory_type: str = "working",
        importance: float = 0.5,
        confidence: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> Optional["MemoryRecord"]:
        """Persist content into the integrated MemoryManager and add an L2 reference."""
        try:
            mm = self._ensure_memory_manager()
        except RuntimeError:
            return None
        record = mm.create_memory(
            memory_type=memory_type,
            content=content,
            importance=importance,
            confidence=confidence,
            tags=tags or [],
        )
        self.integrate_memory(
            memory_id=record.memory_id,
            memory_type=memory_type,
            relevance_score=importance,
        )
        return record

    def request_recruitment(
        self,
        sender_id: str,
        reason: str,
        urgency: float = 0.5,
        min_models: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Emit a recruitment request event and optionally consult the recruiter."""
        event = self.emit(
            CollectiveEventType.RECRUITMENT_REQUEST,
            sender_id=sender_id,
            content=reason,
            confidence=urgency,
            metadata={"min_models": min_models, **(metadata or {})},
        )
        if self._recruiter is not None and self._recruitment_context is not None:
            # Lazy import to avoid heavy deps at module load
            from ..momm.recruitment import RecruitmentContext  # noqa: F401
            # The recruiter expects available_models / active_models lists;
            # callers supply those via the recruitment_context if needed.
            return {
                "event_id": event.event_id,
                "recruiter_strategy": getattr(self._recruiter, "strategy", None),
            }
        return {"event_id": event.event_id}

    def record_failure(
        self,
        model_id: str,
        failure_category: str,
        description: str,
        confidence: float = 0.0,
    ) -> None:
        """Record a failure event for correlated failure detection."""
        if self._failure_detector is not None:
            self._failure_detector.record_failure(
                model_id=model_id,
                task_id=self.task_id,
                failure_category=failure_category,
                description=description,
                confidence=confidence,
            )
        self.emit(
            CollectiveEventType.WARNING,
            sender_id=model_id,
            content=f"Failure: {description}",
            confidence=confidence,
            metadata={"failure_category": failure_category},
        )

    def reflect(
        self,
        model_id: str,
        confidence: float,
        reasoning_quality: float,
        strengths: Optional[List[str]] = None,
        weaknesses: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run self-reflection for a model and record the outcome."""
        if self._self_reflection is None:
            return None
        assessment = self._self_reflection.self_assess(
            model_id=model_id,
            task_id=self.task_id,
            contribution="",
            evidence=[],
            confidence=confidence,
        )
        self.emit(
            CollectiveEventType.SUMMARY,
            sender_id=model_id,
            content=f"Self-assessment: confidence={confidence:.2f}, quality={reasoning_quality:.2f}",
            confidence=confidence,
            metadata={"assessment": getattr(assessment, "to_dict", lambda: {})()},
        )
        return getattr(assessment, "to_dict", lambda: {})()

    def verify_candidate(
        self,
        candidate_id: str,
        content: str,
        evidence: List[str],
        verifier_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Verify a candidate using the integrated verifier stack."""
        result: Optional[Dict[str, Any]] = None
        if self._recursive_verifier is not None:
            def _verification_fn(candidate_content: str, verifier: str) -> Any:
                from ..training.verification import VerificationResult, VerificationOutcome
                return VerificationResult(
                    verification_id=str(uuid.uuid4()),
                    candidate_id=candidate_id,
                    verifier_id=verifier,
                    outcome=VerificationOutcome.PASS if candidate_content == content else VerificationOutcome.FAIL,
                    confidence=0.9,
                    evidence=evidence,
                    assumptions_checked=[],
                    contradictions_found=[],
                    tool_verifications=[],
                    revision_suggestions=[],
                    metadata={},
                    timestamp=datetime.now().isoformat(),
                )
            verification = self._recursive_verifier.verify(
                candidate_id=candidate_id,
                content=content,
                verifier_id=verifier_id,
                verification_fn=_verification_fn,
            )
            result = verification.to_dict() if hasattr(verification, "to_dict") else {}
            self.emit(
                CollectiveEventType.VERIFICATION,
                sender_id=verifier_id,
                content=f"Verified {candidate_id}: {getattr(verification, 'outcome', 'unknown')}",
                confidence=getattr(verification, "confidence", 0.0),
                evidence_refs=evidence,
                metadata={"candidate_id": candidate_id, "verification": result},
            )
        return result

    def meta_verify(
        self,
        candidate_id: str,
        content: str,
        evidence: List[str],
        prior_verifications: List[Any],
        verifier_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Meta-verify a candidate using the integrated meta-verifier."""
        if self._meta_verifier is None:
            return None
        def _meta_verification_fn(content: str, evidence: List[str], verifications: List[Any]) -> Any:
            from ..training.verification import VerificationResult, VerificationOutcome
            return VerificationResult(
                verification_id=str(uuid.uuid4()),
                candidate_id=candidate_id,
                verifier_id=verifier_id,
                outcome=VerificationOutcome.PASS,
                confidence=0.95,
                evidence=evidence,
                assumptions_checked=[],
                contradictions_found=[],
                tool_verifications=[],
                revision_suggestions=[],
                metadata={},
                timestamp=datetime.now().isoformat(),
            )
        result = self._meta_verifier.meta_verify(
            candidate_id=candidate_id,
            content=content,
            evidence=evidence,
            verifications=prior_verifications,
            verifier_id=verifier_id,
            meta_verification_fn=_meta_verification_fn,
        )
        return result.to_dict() if hasattr(result, "to_dict") else None

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """Return workspace statistics."""
        with self._lock:
            return {
                "task_id": self.task_id,
                "session_id": self.session_id,
                "l0_event_count": len(self._events),
                "l1_hypothesis_count": len(self._l1.active_hypotheses),
                "l1_disagreement_count": len(self._l1.disagreements),
                "l1_warning_count": len(self._l1.warnings),
                "l1_open_questions": len(self._l1.open_questions),
                "l1_recruitment_requests": len(self._l1.recruitment_requests),
                "l1_stop_requests": len(self._l1.stop_requests),
                "l1_consensus_confidence": self._l1.consensus_confidence,
                "l2_ref_count": len(self._l2.references),
                "active_slots": len(self.get_active_slots()),
                "snapshot_count": len(self._snapshots),
                "default_token_budget": self.default_token_budget,
            }

    def get_layer0_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a compact L0 summary."""
        with self._lock:
            return [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type,
                    "sender_id": e.sender_id,
                    "content": e.content[:120],
                    "confidence": e.confidence,
                    "timestamp": e.timestamp,
                }
                for e in self._events[-limit:]
            ]

    def get_layer1_summary(self) -> Dict[str, Any]:
        """Return a compact L1 summary."""
        with self._lock:
            return {
                "consensus_candidate": self._l1.consensus_candidate,
                "consensus_confidence": self._l1.consensus_confidence,
                "active_hypotheses": len(self._l1.active_hypotheses),
                "disagreements": len(self._l1.disagreements),
                "warnings": len(self._l1.warnings),
                "open_questions": len(self._l1.open_questions),
                "recruitment_requests": len(self._l1.recruitment_requests),
                "stop_requests": len(self._l1.stop_requests),
                "has_summary": self._l1.summary is not None,
                "updated_at": self._l1.updated_at,
            }

    def get_layer2_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return a compact L2 summary."""
        with self._lock:
            refs = sorted(
                self._l2.references.values(),
                key=lambda r: (r.relevance_score, r.last_accessed),
                reverse=True,
            )[:limit]
            return [
                {
                    "ref_id": r.ref_id,
                    "memory_id": r.memory_id,
                    "memory_type": r.memory_type,
                    "relevance_score": r.relevance_score,
                    "access_count": r.access_count,
                    "last_accessed": r.last_accessed,
                }
                for r in refs
            ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _persist_if_enabled(self) -> None:
        """Persist state to disk if persistence is enabled."""
        if not self.persistence_enabled or self._state_path is None:
            return
        try:
            self._save_state()
        except Exception:
            pass  # Best-effort persistence

    def _save_state(self) -> None:
        """Save workspace state to disk."""
        if self._state_path is None:
            return
        data = {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "default_token_budget": self.default_token_budget,
            "max_events": self.max_events,
            "max_slots": self.max_slots,
            "max_l2_refs": self.max_l2_refs,
            "events": [e.to_dict() for e in self._events],
            "l1_state": self._l1.to_dict(),
            "l2_context": self._l2.to_dict(),
            "slots": {sid: s.to_dict() for sid, s in self._slots.items()},
            "snapshots": {k: v.to_dict() for k, v in self._snapshots.items()},
            "saved_at": datetime.now().isoformat(),
        }
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._state_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        tmp_path.replace(self._state_path)

    def _load_state(self) -> None:
        """Load workspace state from disk."""
        if self._state_path is None or not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r") as f:
                data = json.load(f)
            self._events = [CollectiveEvent.from_dict(e) for e in data.get("events", [])]
            self._l1 = CollectiveWorkingState.from_dict(data.get("l1_state", {}))
            self._l2 = LongTermMemoryContext.from_dict(data.get("l2_context", {}))
            self._slots = {sid: ContextSlot.from_dict(s) for sid, s in data.get("slots", {}).items()}
            self._model_slot_index = {}
            for slot in self._slots.values():
                if slot.is_active():
                    self._model_slot_index[slot.model_id] = slot.slot_id
            self._snapshots = {
                k: WorkspaceSnapshot.from_dict(v) for k, v in data.get("snapshots", {}).items()
            }
            cfg = data.get("token_budget_config", {})
            if cfg:
                self.default_token_budget = cfg.get("default_token_budget", self.default_token_budget)
                self.max_events = cfg.get("max_events", self.max_events)
                self.max_slots = cfg.get("max_slots", self.max_slots)
                self.max_l2_refs = cfg.get("max_l2_refs", self.max_l2_refs)
        except Exception:
            pass  # Corrupted state file; start fresh

    def serialize(self) -> Dict[str, Any]:
        """Serialise the full workspace state."""
        with self._lock:
            return {
                "task_id": self.task_id,
                "session_id": self.session_id,
                "default_token_budget": self.default_token_budget,
                "max_events": self.max_events,
                "max_slots": self.max_slots,
                "max_l2_refs": self.max_l2_refs,
                "events": [e.to_dict() for e in self._events],
                "l1_state": self._l1.to_dict(),
                "l2_context": self._l2.to_dict(),
                "slots": {sid: s.to_dict() for sid, s in self._slots.items()},
                "snapshots": {k: v.to_dict() for k, v in self._snapshots.items()},
            }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "CollectiveReasoningWorkspace":
        """Reconstruct a workspace from a serialised dict."""
        ws = cls(
            task_id=data["task_id"],
            session_id=data["session_id"],
            default_token_budget=data.get("default_token_budget", 4096),
            max_events=data.get("max_events", 10000),
            max_slots=data.get("max_slots", 32),
            max_l2_refs=data.get("max_l2_refs", 5000),
            persistence_enabled=False,
        )
        ws._events = [CollectiveEvent.from_dict(e) for e in data.get("events", [])]
        ws._l1 = CollectiveWorkingState.from_dict(data.get("l1_state", {}))
        ws._l2 = LongTermMemoryContext.from_dict(data.get("l2_context", {}))
        ws._slots = {sid: ContextSlot.from_dict(s) for sid, s in data.get("slots", {}).items()}
        ws._model_slot_index = {}
        for slot in ws._slots.values():
            if slot.is_active():
                ws._model_slot_index[slot.model_id] = slot.slot_id
        ws._snapshots = {
            k: WorkspaceSnapshot.from_dict(v) for k, v in data.get("snapshots", {}).items()
        }
        return ws

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def clear(self) -> None:
        """Reset the workspace to an empty state (keeps config)."""
        with self._lock:
            self._events.clear()
            self._l1 = CollectiveWorkingState(task_id=self.task_id, session_id=self.session_id)
            self._l2 = LongTermMemoryContext(task_id=self.task_id, session_id=self.session_id)
            self._slots.clear()
            self._model_slot_index.clear()
            self._snapshots.clear()
            self._persist_if_enabled()

    def close(self) -> None:
        """Release all resources and persist final state."""
        with self._lock:
            for model_id in list(self._model_slot_index.keys()):
                self.release_slot(model_id)
            self._persist_if_enabled()
