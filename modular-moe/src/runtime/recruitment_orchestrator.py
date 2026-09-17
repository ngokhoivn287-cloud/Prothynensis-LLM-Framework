"""Runtime recruitment state and orchestrator for Prothynesis.

Implements the full dynamic recruitment loop:

task -> task analysis -> capability requirements -> initial candidate retrieval
-> reputation filtering -> information-gain estimation -> initial recruitment
-> parallel reasoning -> uncertainty / disagreement analysis -> dynamic recruitment
-> verification -> stop recruitment when additional models have low expected value
-> synthesis
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

from momm.recruitment import DynamicRecruiter, RecruitmentContext, ActivationBudget, RecruitmentBudget, RecruitmentStrategy


class RecruitmentDecision(Enum):
    RECRUIT = "recruit"
    STOP = "stop"
    VERIFY = "verify"
    SYNTHESIZE = "synthesize"


class RecruitmentStopReason(Enum):
    LOW_INFORMATION_GAIN = "low_information_gain"
    LOW_UNCERTAINTY = "low_uncertainty"
    LOW_DISAGREEMENT = "low_disagreement"
    COMPUTE_BUDGET_EXHAUSTED = "compute_budget_exhausted"
    LATENCY_BUDGET_EXHAUSTED = "latency_budget_exhausted"
    MAX_ACTIVE_REACHED = "max_active_reached"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    TARGET_ACCURACY_REACHED = "target_accuracy_reached"


@dataclass
class CandidateOutput:
    """Output from a recruited model."""
    model_id: str
    content: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    uncertainty: str = "not_verified"
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecruitmentRound:
    """One round of recruitment and execution."""
    round_id: int
    recruited_models: List[str] = field(default_factory=list)
    outputs: List[CandidateOutput] = field(default_factory=list)
    uncertainty: float = 0.0
    disagreement: float = 0.0
    expected_information_gain: float = 0.0
    active_parameters: int = 0
    decision: str = RecruitmentDecision.RECRUIT.value
    stop_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outputs"] = [o.to_dict() for o in self.outputs]
        return d


@dataclass
class RecruitmentState:
    """Full runtime recruitment state for a task."""
    task_id: str
    current_round: int = 0
    max_rounds: int = 3
    max_recruitments_per_task: int = 5
    candidate_pool: List[str] = field(default_factory=list)
    available_models: List[str] = field(default_factory=list)
    active_models: List[str] = field(default_factory=list)
    recruited_models: List[str] = field(default_factory=list)
    rounds: List[RecruitmentRound] = field(default_factory=list)
    capability_requirements: Dict[str, float] = field(default_factory=dict)
    uncertainty_history: List[float] = field(default_factory=list)
    disagreement_history: List[float] = field(default_factory=list)
    information_gain_history: List[float] = field(default_factory=list)
    compute_budget_remaining: float = 1.0
    latency_budget_remaining: float = 60.0
    recruitment_reason: str = ""
    stop_reason: Optional[str] = None
    final_outputs: List[CandidateOutput] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["rounds"] = [r.to_dict() for r in self.rounds]
        d["final_outputs"] = [o.to_dict() for o in self.final_outputs]
        return d


class RecruitmentOrchestrator:
    """
    Orchestrates dynamic runtime recruitment.
    
    Manages the full lifecycle:
    1. Task analysis and capability requirement extraction
    2. Initial candidate retrieval and ranking
    3. Initial recruitment with reputation/information-gain filtering
    4. Parallel reasoning execution
    5. Uncertainty/disagreement analysis
    6. Dynamic re-recruitment or stop
    7. Verification
    8. Synthesis handoff
    """

    def __init__(self, config: dict):
        self.config = config
        self.recruiter = DynamicRecruiter(config)
        self.budget = RecruitmentBudget(config)
        self.max_rounds = config.get("recruitment", {}).get("max_rounds", 3)
        self.max_recruitments_per_task = config.get("recruitment", {}).get("max_recruitments_per_task", 5)
        self.min_information_gain = config.get("recruitment", {}).get("min_information_gain", 0.01)
        self.max_active_models = config.get("recruitment", {}).get("max_active_models", 32)
        self.uncertainty_threshold = config.get("recruitment", {}).get("uncertainty_threshold", 0.3)
        self.disagreement_threshold = config.get("recruitment", {}).get("disagreement_threshold", 0.15)
        self._states: Dict[str, RecruitmentState] = {}
        self._lock = threading.Lock()

    def start_task(self, task_id: str, available_models: List[Any], task_metadata: Optional[Dict[str, Any]] = None) -> RecruitmentState:
        """Initialize recruitment state for a new task."""
        with self._lock:
            state = RecruitmentState(
                task_id=task_id,
                max_rounds=self.max_rounds,
                max_recruitments_per_task=self.max_recruitments_per_task,
                available_models=[getattr(m, "model_id", str(m)) for m in available_models],
                metadata=task_metadata or {},
            )
            self._states[task_id] = state
            return state

    def get_state(self, task_id: str) -> Optional[RecruitmentState]:
        """Get current recruitment state for a task."""
        with self._lock:
            return self._states.get(task_id)

    def initial_recruitment(
        self,
        task_id: str,
        available_models: List[Any],
        task_requirements: Dict[str, Any],
        capability_requirements: Dict[str, float],
        reputation_store: Any = None,
        information_gain_estimator: Any = None,
    ) -> List[Any]:
        """
        Perform initial recruitment for a task.
        
        Args:
            task_id: Task identifier
            available_models: List of available model objects
            task_requirements: Task-specific requirements (difficulty, domain, etc.)
            capability_requirements: Required capability thresholds
            reputation_store: Optional reputation store for filtering
            information_gain_estimator: Optional information gain estimator
        """
        state = self.get_state(task_id)
        if state is None:
            state = self.start_task(task_id, available_models, task_requirements)

        state.capability_requirements = capability_requirements
        difficulty = task_requirements.get("difficulty", 0.5)
        verification_needed = task_requirements.get("verification_needed", False)

        context = RecruitmentContext(
            task_difficulty=difficulty,
            uncertainty=0.0,
            disagreement=0.0,
            expected_information_gain=0.0,
            verification_needed=verification_needed,
            compute_budget=state.compute_budget_remaining,
            latency_budget=state.latency_budget_remaining,
            current_active_models=0,
            max_active_models=self.max_active_models,
            min_active_models=self.config.get("recruitment", {}).get("min_active_models", 1),
        )

        ranked = self._rank_candidates(
            available_models,
            capability_requirements,
            reputation_store,
            information_gain_estimator,
            state,
        )
        state.candidate_pool = [getattr(m, "model_id", str(m)) for m in ranked]

        selected = self.recruiter.recruit(context, ranked, [])
        selected_ids = [getattr(m, "model_id", str(m)) for m in selected]
        state.active_models = selected_ids
        state.recruited_models.extend(selected_ids)

        activation = self.recruiter.estimate_activation(selected)
        state.rounds.append(RecruitmentRound(
            round_id=state.current_round,
            recruited_models=selected_ids,
            uncertainty=0.0,
            disagreement=0.0,
            expected_information_gain=0.0,
            active_parameters=activation.active_parameters,
            decision=RecruitmentDecision.RECRUIT.value,
            metadata={"initial_recruitment": True, "difficulty": difficulty},
        ))
        state.current_round += 1
        state.updated_at = time.time()
        return selected

    def evaluate_and_recruit(
        self,
        task_id: str,
        outputs: List[CandidateOutput],
        uncertainty: float,
        disagreement: float,
        available_models: List[Any],
        capability_requirements: Dict[str, float],
        reputation_store: Any = None,
        information_gain_estimator: Any = None,
        verification_fn: Optional[Callable[[str], Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate current round and decide whether to recruit more or stop.
        
        Returns:
            dict with keys:
                decision: "recruit" | "stop" | "verify" | "synthesize"
                recruited: list of newly recruited model objects
                stop_reason: optional stop reason
                new_round: RecruitmentRound
        """
        state = self.get_state(task_id)
        if state is None:
            return {"decision": RecruitmentDecision.STOP.value, "stop_reason": "no_state", "recruited": []}

        info_gain = self._estimate_round_information_gain(
            outputs, state, information_gain_estimator
        )
        state.information_gain_history.append(info_gain)
        state.uncertainty_history.append(uncertainty)
        state.disagreement_history.append(disagreement)

        stop_decision = self._should_stop_recruitment(
            state, uncertainty, disagreement, info_gain, len(available_models)
        )
        if stop_decision is not None:
            decision, stop_reason = stop_decision
            state.stop_reason = stop_reason.value if hasattr(stop_reason, "value") else str(stop_reason)
            state.final_outputs.extend(outputs)
            round_entry = RecruitmentRound(
                round_id=state.current_round,
                recruited_models=[],
                outputs=outputs,
                uncertainty=uncertainty,
                disagreement=disagreement,
                expected_information_gain=info_gain,
                decision=decision.value if hasattr(decision, "value") else str(decision),
                stop_reason=state.stop_reason,
                metadata={"rounds_exhausted": state.current_round >= state.max_rounds},
            )
            state.rounds.append(round_entry)
            state.updated_at = time.time()
            return {
                "decision": decision.value if hasattr(decision, "value") else str(decision),
                "stop_reason": state.stop_reason,
                "recruited": [],
                "new_round": round_entry,
            }

        if uncertainty > self.uncertainty_threshold or disagreement > self.disagreement_threshold:
            decision = RecruitmentDecision.RECRUIT
        elif verification_fn is not None and outputs:
            decision = RecruitmentDecision.VERIFY
        else:
            decision = RecruitmentDecision.SYNTHESIZE

        ranked = self._rank_candidates(
            available_models,
            state.capability_requirements,
            reputation_store,
            information_gain_estimator,
            state,
        )
        state.candidate_pool = [getattr(m, "model_id", str(m)) for m in ranked]

        context = RecruitmentContext(
            task_difficulty=state.metadata.get("difficulty", 0.5),
            uncertainty=uncertainty,
            disagreement=disagreement,
            expected_information_gain=info_gain,
            verification_needed=decision == RecruitmentDecision.VERIFY,
            compute_budget=state.compute_budget_remaining,
            latency_budget=state.latency_budget_remaining,
            current_active_models=len(state.active_models),
            max_active_models=self.max_active_models,
            min_active_models=self.config.get("recruitment", {}).get("min_active_models", 1),
        )

        selected = self.recruiter.recruit(context, ranked, [])
        selected_ids = [getattr(m, "model_id", str(m)) for m in selected]
        new_ids = [mid for mid in selected_ids if mid not in state.recruited_models]
        state.active_models = selected_ids
        state.recruited_models.extend(new_ids)

        activation = self.recruiter.estimate_activation(selected)
        round_entry = RecruitmentRound(
            round_id=state.current_round,
            recruited_models=new_ids,
            outputs=outputs,
            uncertainty=uncertainty,
            disagreement=disagreement,
            expected_information_gain=info_gain,
            active_parameters=activation.active_parameters,
            decision=decision.value if hasattr(decision, "value") else str(decision),
            metadata={"info_gain": info_gain, "uncertainty": uncertainty, "disagreement": disagreement},
        )
        state.rounds.append(round_entry)
        state.current_round += 1
        state.updated_at = time.time()
        return {
            "decision": decision.value if hasattr(decision, "value") else str(decision),
            "recruited": selected,
            "new_ids": new_ids,
            "stop_reason": None,
            "new_round": round_entry,
        }

    def _rank_candidates(
        self,
        available_models: List[Any],
        capability_requirements: Dict[str, float],
        reputation_store: Any,
        information_gain_estimator: Any,
        state: RecruitmentState,
    ) -> List[Any]:
        candidates = list(available_models)
        if reputation_store is not None:
            def _rep_score(m: Any) -> float:
                mid = getattr(m, "model_id", str(m))
                rep = reputation_store.get_reputation(mid)
                if rep is None:
                    return 0.5
                return rep.reliability_score
            candidates.sort(key=_rep_score, reverse=True)

        if information_gain_estimator is not None and len(state.recruited_models) > 0:
            population_capabilities = {}
            for m in available_models:
                mid = getattr(m, "model_id", str(m))
                cv = getattr(m, "capability_vector", {})
                if hasattr(cv, "__dataclass_fields__"):
                    cv = {f: getattr(cv, f, 0.0) for f in cv.__dataclass_fields__}
                elif not isinstance(cv, dict):
                    cv = {}
                population_capabilities[mid] = list(cv.values())

            def _info_gain(m: Any) -> float:
                mid = getattr(m, "model_id", str(m))
                if mid in state.recruited_models:
                    return -1.0
                cv = getattr(m, "capability_vector", {})
                if hasattr(cv, "__dataclass_fields__"):
                    cv = {f: getattr(cv, f, 0.0) for f in cv.__dataclass_fields__}
                elif not isinstance(cv, dict):
                    cv = {}
                return information_gain_estimator.estimate_gain(
                    model_id=mid,
                    capability_vector=cv,
                    reliability=_rep_score(m) if reputation_store is not None else 0.5,
                    population_capabilities=population_capabilities,
                ).composite_gain
            candidates.sort(key=_info_gain, reverse=True)
        return candidates

    def _estimate_round_information_gain(
        self,
        outputs: List[CandidateOutput],
        state: RecruitmentState,
        information_gain_estimator: Any,
    ) -> float:
        if not outputs:
            return 0.0
        current_contents = list({o.content for o in outputs})
        if not state.rounds:
            return 0.5
        previous_contents = list({o.content for r in state.rounds for o in r.outputs})
        if not previous_contents:
            return 0.5
        new_fraction = len([c for c in current_contents if c not in previous_contents]) / max(1, len(current_contents))
        return max(0.0, min(1.0, new_fraction))

    def _should_stop_recruitment(
        self,
        state: RecruitmentState,
        uncertainty: float,
        disagreement: float,
        info_gain: float,
        remaining_available: int,
    ) -> Optional[tuple]:
        if state.current_round >= state.max_rounds:
            return RecruitmentDecision.STOP, RecruitmentStopReason.MAX_ROUNDS_REACHED
        if state.current_round >= state.max_recruitments_per_task:
            return RecruitmentDecision.STOP, RecruitmentStopReason.MAX_ROUNDS_REACHED
        if info_gain < self.min_information_gain and uncertainty < self.uncertainty_threshold:
            return RecruitmentDecision.STOP, RecruitmentStopReason.LOW_INFORMATION_GAIN
        if uncertainty < self.uncertainty_threshold * 0.5 and disagreement < self.disagreement_threshold * 0.5:
            return RecruitmentDecision.STOP, RecruitmentStopReason.LOW_UNCERTAINTY
        if disagreement < self.disagreement_threshold * 0.5 and info_gain < self.min_information_gain:
            return RecruitmentDecision.STOP, RecruitmentStopReason.LOW_DISAGREEMENT
        if len(state.active_models) >= self.max_active_models:
            return RecruitmentDecision.STOP, RecruitmentStopReason.MAX_ACTIVE_REACHED
        if state.compute_budget_remaining <= 0:
            return RecruitmentDecision.STOP, RecruitmentStopReason.COMPUTE_BUDGET_EXHAUSTED
        if state.latency_budget_remaining <= 0:
            return RecruitmentDecision.STOP, RecruitmentStopReason.LATENCY_BUDGET_EXHAUSTED
        return None

    def get_recruitment_trace(self, task_id: str) -> List[Dict[str, Any]]:
        """Get structured recruitment trace for a task."""
        state = self.get_state(task_id)
        if state is None:
            return []
        return [r.to_dict() for r in state.rounds]

    def get_active_parameter_count(self, task_id: str) -> int:
        """Get current active parameter count for a task."""
        state = self.get_state(task_id)
        if state is None or not state.rounds:
            return 0
        return state.rounds[-1].active_parameters

    def finalize_task(self, task_id: str, final_outputs: List[CandidateOutput]) -> Optional[RecruitmentState]:
        """Finalize recruitment and store final outputs."""
        with self._lock:
            state = self._states.get(task_id)
            if state is None:
                return None
            state.final_outputs = final_outputs
            state.updated_at = time.time()
            return state
