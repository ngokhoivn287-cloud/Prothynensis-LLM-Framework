"""Runtime inference orchestrator with dynamic recruitment for Prothynesis.

Wires RecruitmentOrchestrator into the inference path:

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
from typing import Dict, List, Optional, Any, Callable, Union, Iterator
from pathlib import Path

try:
    from momm.inference import InferenceEngine, InferenceConfig, ModelCache
    from momm.registry import ModelRegistry, create_registry, ModelMetadata
    from momm.coordinator import ModelCoordinator, CoordinatorConfig
    from momm.collaboration import CollaborativeSynthesizer, CandidateResult, SynthesisStrategy
    from momm.communication import PeerCommunicationProtocol, MessageType
    from momm.memory import MemoryManager, MemoryRecord, MemoryType
except ImportError:
    try:
        from src.momm.inference import InferenceEngine, InferenceConfig, ModelCache
        from src.momm.registry import ModelRegistry, create_registry, ModelMetadata
        from src.momm.coordinator import ModelCoordinator, CoordinatorConfig
        from src.momm.collaboration import CollaborativeSynthesizer, CandidateResult, SynthesisStrategy
        from src.momm.communication import PeerCommunicationProtocol, MessageType
        from src.momm.memory import MemoryManager, MemoryRecord, MemoryType
    except ImportError:
        InferenceEngine = None  # type: ignore[misc,assignment]
        InferenceConfig = None  # type: ignore[misc,assignment]
        ModelCache = None  # type: ignore[misc,assignment]
        ModelRegistry = None  # type: ignore[misc,assignment]
        create_registry = None  # type: ignore[assignment]
        ModelMetadata = None  # type: ignore[misc,assignment]
        ModelCoordinator = None  # type: ignore[misc,assignment]
        CoordinatorConfig = None  # type: ignore[misc,assignment]
        CollaborativeSynthesizer = None  # type: ignore[misc,assignment]
        CandidateResult = None  # type: ignore[misc,assignment]
        SynthesisStrategy = None  # type: ignore[misc,assignment]
        PeerCommunicationProtocol = None  # type: ignore[misc,assignment]
        MessageType = None  # type: ignore[misc,assignment]
        MemoryManager = None  # type: ignore[misc,assignment]
        MemoryRecord = None  # type: ignore[misc,assignment]
        MemoryType = None  # type: ignore[misc,assignment]
try:
    from training.reputation import ReputationStore
    from training.calibration import UncertaintyEstimator, UncertaintyLevel
    from training.information_gain import InformationGainEstimator
    from training.verification import RecursiveVerifier, MetaVerifier, VerificationOutcome
except ImportError:
    try:
        from src.training.reputation import ReputationStore
        from src.training.calibration import UncertaintyEstimator, UncertaintyLevel
        from src.training.information_gain import InformationGainEstimator
        from src.training.verification import RecursiveVerifier, MetaVerifier, VerificationOutcome
    except ImportError:
        ReputationStore = None  # type: ignore[misc,assignment]
        UncertaintyEstimator = None  # type: ignore[misc,assignment]
        UncertaintyLevel = None  # type: ignore[misc,assignment]
        InformationGainEstimator = None  # type: ignore[misc,assignment]
        RecursiveVerifier = None  # type: ignore[misc,assignment]
        MetaVerifier = None  # type: ignore[misc,assignment]
        VerificationOutcome = None  # type: ignore[misc,assignment]
from .recruitment_orchestrator import (
    RecruitmentOrchestrator,
    RecruitmentState,
    RecruitmentRound,
    CandidateOutput,
    RecruitmentDecision,
    RecruitmentStopReason,
)


@dataclass
class TaskContext:
    """Context for a runtime task."""
    task_id: str
    prompt: str
    difficulty: float = 0.5
    domain: str = "general"
    task_type: str = "reasoning"
    verification_needed: bool = False
    expected_outputs: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CollaborationGain:
    """Records collaboration gain metrics."""
    task_id: str
    single_model_accuracy: float = 0.0
    population_accuracy: float = 0.0
    expanded_population_accuracy: float = 0.0
    single_model_uncertainty: float = 0.0
    population_uncertainty: float = 0.0
    expanded_population_uncertainty: float = 0.0
    active_parameters_single: int = 0
    active_parameters_population: int = 0
    active_parameters_expanded: int = 0
    latency_single: float = 0.0
    latency_population: float = 0.0
    latency_expanded: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class RuntimeInferenceOrchestrator:
    """
    Orchestrates runtime inference with dynamic recruitment.
    
    Integrates:
    - RecruitmentOrchestrator for dynamic model selection
    - InferenceEngine for model execution
    - CollaborativeSynthesizer for synthesis
    - PeerCommunicationProtocol for model communication
    - MemoryManager for working memory
    - ReputationStore for model reliability
    - UncertaintyEstimator for uncertainty tracking
    - InformationGainEstimator for recruitment value estimation
    - RecursiveVerifier and MetaVerifier for verification
    """

    def __init__(
        self,
        config: Dict[str, Any],
        registry: Optional[ModelRegistry] = None,
        inference_engine: Optional[InferenceEngine] = None,
        reputation_store: Optional[ReputationStore] = None,
        inference_fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.config = config
        self.registry = registry or create_registry(
            registry_dir=config.get("runtime", {}).get("registry_dir", "models"),
            max_gpu_models=config.get("runtime", {}).get("max_gpu_models", 2),
            max_cpu_models=config.get("runtime", {}).get("max_cpu_models", 8),
            lazy_loading=config.get("runtime", {}).get("lazy_loading", True),
            cpu_offload=config.get("runtime", {}).get("cpu_offload", True),
        )
        self.inference_engine = inference_engine or InferenceEngine(
            config=InferenceConfig(
                max_gpu_models=config.get("runtime", {}).get("max_gpu_models", 2),
                max_cpu_models=config.get("runtime", {}).get("max_cpu_models", 8),
                lazy_loading=config.get("runtime", {}).get("lazy_loading", True),
                cpu_offload=config.get("runtime", {}).get("cpu_offload", True),
                compile_models=config.get("runtime", {}).get("compile_models", False),
                use_flash_attn=config.get("runtime", {}).get("use_flash_attn", False),
                default_top_k=config.get("generation", {}).get("default_top_k", 2),
                default_temperature=config.get("generation", {}).get("default_temperature", 1.0),
                default_top_p=config.get("generation", {}).get("default_top_p", 1.0),
                default_max_new_tokens=config.get("generation", {}).get("default_max_new_tokens", 100),
                default_repetition_penalty=config.get("generation", {}).get("default_repetition_penalty", 1.0),
            ),
            registry=self.registry,
        )
        self.recruitment_orchestrator = RecruitmentOrchestrator(config)
        self.reputation_store = reputation_store or ReputationStore(
            storage_path=config.get("reputation", {}).get("storage_path")
        )
        self.uncertainty_estimator = UncertaintyEstimator(config)
        self.information_gain_estimator = InformationGainEstimator(config)
        self.recursive_verifier = RecursiveVerifier(config)
        self.meta_verifier = MetaVerifier(config)
        self.inference_fn = inference_fn
        self._round_counter = 0
        self._task_memories: Dict[str, MemoryManager] = {}
        self._task_protocols: Dict[str, PeerCommunicationProtocol] = {}
        self._lock = threading.Lock()

    def execute_task(
        self,
        task_context: TaskContext,
        available_models: List[Any],
        capability_requirements: Optional[Dict[str, float]] = None,
        verification_fn: Optional[Callable[[str], Any]] = None,
        synthesis_fn: Optional[Callable[[List[CandidateOutput]], str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task with dynamic recruitment.
        
        Args:
            task_context: Task context
            available_models: Available model objects
            capability_requirements: Required capability thresholds
            verification_fn: Optional verification function(content) -> VerificationResult
            synthesis_fn: Optional synthesis function(outputs) -> final answer
        
        Returns:
            dict with final_output, recruitment_trace, collaboration_gain, etc.
        """
        start_time = time.time()
        task_id = task_context.task_id
        capability_requirements = capability_requirements or {}

        state = self.recruitment_orchestrator.start_task(
            task_id, available_models, task_context.to_dict()
        )

        memory = MemoryManager(
            model_id="orchestrator",
            task_id=task_id,
            session_id=task_id,
        )
        protocol = PeerCommunicationProtocol(model_id="orchestrator", task_id=task_id)
        self._task_memories[task_id] = memory
        self._task_protocols[task_id] = protocol

        selected = self.recruitment_orchestrator.initial_recruitment(
            task_id=task_id,
            available_models=available_models,
            task_requirements=task_context.to_dict(),
            capability_requirements=capability_requirements,
            reputation_store=self.reputation_store,
            information_gain_estimator=self.information_gain_estimator,
        )
        selected_ids = [getattr(m, "model_id", str(m)) for m in selected]

        outputs = self._execute_models(task_context, selected)
        initial_activation = self.recruitment_orchestrator.recruiter.estimate_activation(selected)

        uncertainty = self._compute_uncertainty(outputs)
        disagreement = self._compute_disagreement(outputs)
        collaboration_gain = CollaborationGain(
            task_id=task_id,
            single_model_accuracy=outputs[0].confidence if outputs else 0.0,
            population_accuracy=sum(o.confidence for o in outputs) / max(1, len(outputs)),
            active_parameters_single=initial_activation.active_parameters // max(1, len(selected)),
            active_parameters_population=initial_activation.active_parameters,
            latency_single=sum(o.metadata.get("duration", 0.0) for o in outputs[:1]),
            latency_population=sum(o.metadata.get("duration", 0.0) for o in outputs),
        )

        eval_result = self.recruitment_orchestrator.evaluate_and_recruit(
            task_id=task_id,
            outputs=outputs,
            uncertainty=uncertainty,
            disagreement=disagreement,
            available_models=available_models,
            capability_requirements=capability_requirements,
            reputation_store=self.reputation_store,
            information_gain_estimator=self.information_gain_estimator,
            verification_fn=verification_fn,
        )

        all_outputs = list(outputs)
        recruited_ids = []
        while eval_result.get("decision") == RecruitmentDecision.RECRUIT.value:
            new_models = eval_result.get("recruited", [])
            new_ids = [getattr(m, "model_id", str(m)) for m in new_models]
            recruited_ids.extend(new_ids)
            new_outputs = self._execute_models(task_context, new_models)
            all_outputs.extend(new_outputs)

            uncertainty = self._compute_uncertainty(all_outputs)
            disagreement = self._compute_disagreement(all_outputs)
            eval_result = self.recruitment_orchestrator.evaluate_and_recruit(
                task_id=task_id,
                outputs=all_outputs,
                uncertainty=uncertainty,
                disagreement=disagreement,
                available_models=[m for m in available_models if getattr(m, "model_id", str(m)) not in state.recruited_models],
                capability_requirements=capability_requirements,
                reputation_store=self.reputation_store,
                information_gain_estimator=self.information_gain_estimator,
                verification_fn=verification_fn,
            )

        if verification_fn is not None and eval_result.get("decision") == RecruitmentDecision.VERIFY.value:
            verified_outputs = []
            for o in all_outputs:
                vr = verification_fn(o.content)
                o.metadata["verification"] = vr.to_dict() if hasattr(vr, "to_dict") else str(vr)
                verified_outputs.append(o)
            all_outputs = verified_outputs
            uncertainty = self._compute_uncertainty(all_outputs)
            disagreement = self._compute_disagreement(all_outputs)

        if synthesis_fn is not None:
            final_output = synthesis_fn(all_outputs)
        else:
            final_output = self._default_synthesis(all_outputs)

        final_state = self.recruitment_orchestrator.finalize_task(task_id, all_outputs)
        if final_state is not None and final_state.rounds:
            final_activation = final_state.rounds[-1].active_parameters
        else:
            final_activation = initial_activation.active_parameters

        collaboration_gain.expanded_population_accuracy = sum(o.confidence for o in all_outputs) / max(1, len(all_outputs))
        collaboration_gain.expanded_population_uncertainty = uncertainty
        collaboration_gain.active_parameters_expanded = final_activation
        collaboration_gain.latency_expanded = time.time() - start_time

        for o in all_outputs:
            rep = self.reputation_store.get_or_create(o.model_id, getattr(o, "model_family", "general_solver"))
            rep.contribution_ledger.record_participation(contributed=(o in all_outputs[:1]))
            self.reputation_store.update_reputation(rep)

        return {
            "task_id": task_id,
            "final_output": final_output,
            "candidate_outputs": [o.to_dict() for o in all_outputs],
            "recruitment_trace": self.recruitment_orchestrator.get_recruitment_trace(task_id),
            "collaboration_gain": collaboration_gain.to_dict(),
            "uncertainty": uncertainty,
            "disagreement": disagreement,
            "active_parameters": final_activation,
            "selected_models": selected_ids,
            "recruited_models": recruited_ids,
            "latency": time.time() - start_time,
            "stop_reason": final_state.stop_reason if final_state else None,
            "rounds": final_state.current_round if final_state else 0,
        }

    def _execute_models(self, task_context: TaskContext, models: List[Any]) -> List[CandidateOutput]:
        self._round_counter += 1
        current_round = self._round_counter
        outputs = []
        for model in models:
            model_id = getattr(model, "model_id", str(model))
            start = time.time()
            try:
                if self.inference_fn is not None:
                    content = self.inference_fn(task_context.prompt, model_id, current_round)
                else:
                    content = self.inference_engine.generate(task_context.prompt)
                    if hasattr(content, "__iter__") and not isinstance(content, str):
                        content = "".join(content)
            except Exception as e:
                content = f"ERROR: {e}"
            duration = time.time() - start
            confidence = 0.5
            if hasattr(model, "capability_vector"):
                cv = model.capability_vector
                if hasattr(cv, "__dataclass_fields__"):
                    confidence = sum(getattr(cv, f, 0.0) for f in cv.__dataclass_fields__) / max(1, len(cv.__dataclass_fields__))
                elif hasattr(cv, "__dict__"):
                    confidence = sum(cv.__dict__.values()) / max(1, len(cv.__dict__))
            output = CandidateOutput(
                model_id=model_id,
                content=content,
                confidence=confidence,
                metadata={"duration": duration, "task_id": task_context.task_id, "round": current_round},
            )
            outputs.append(output)
        return outputs

    def _compute_uncertainty(self, outputs: List[CandidateOutput]) -> float:
        if not outputs:
            return 0.0
        confidences = [o.confidence for o in outputs]
        if len(confidences) <= 1:
            return 1.0 - confidences[0]
        avg_conf = sum(confidences) / len(confidences)
        variance = sum((c - avg_conf) ** 2 for c in confidences) / len(confidences)
        return min(1.0, variance + (1.0 - avg_conf))

    def _compute_disagreement(self, outputs: List[CandidateOutput]) -> float:
        if len(outputs) <= 1:
            return 0.0
        contents = [o.content for o in outputs]
        total_pairs = 0
        disagree = 0
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                total_pairs += 1
                if contents[i] != contents[j]:
                    disagree += 1
        if total_pairs == 0:
            return 0.0
        return disagree / total_pairs

    def _default_synthesis(self, outputs: List[CandidateOutput]) -> str:
        if not outputs:
            return ""
        best = max(outputs, key=lambda o: o.confidence)
        return best.content

    def get_recruitment_trace(self, task_id: str) -> List[Dict[str, Any]]:
        return self.recruitment_orchestrator.get_recruitment_trace(task_id)

    def get_active_parameter_count(self, task_id: str) -> int:
        return self.recruitment_orchestrator.get_active_parameter_count(task_id)

    def cleanup_task(self, task_id: str) -> None:
        """Clean up task-scoped state."""
        with self._lock:
            self._task_memories.pop(task_id, None)
            self._task_protocols.pop(task_id, None)
