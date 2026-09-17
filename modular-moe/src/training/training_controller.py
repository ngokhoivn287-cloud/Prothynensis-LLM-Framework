"""Centralized training controller for Prothynesis population training.

Responsibilities:
- Allocate training budgets
- Allocate datasets
- Monitor quality and time
- Trigger restarts
- Select best checkpoints
- Trigger distillation, hard-example replay, curriculum changes
- Protect diversity
- Enforce quality-first training policy
"""

from __future__ import annotations

import time
import threading
import math
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

from .base_trainer import BaseTrainer
from .multi_run import MultiRunTrainer, RunMetrics, MultiRunResult
from .adaptive_budget import AdaptiveBudgetAllocator, BudgetAllocation, TrainingBudget, BudgetPhase, StopReason
from .quality_lock import QualityLock, QualityProfile, QualityTier
from .population_scheduler import PopulationScheduler, ModelTrainingRecord, ModelTrainingStatus
from .evolution import PopulationEvolution, GenerationManifest, ModelFitness
from .hard_examples import HardExampleMiner, CurriculumUpdater, FailureLog
from .distillation import DistillationLoss, TeacherConfig
from .checkpoint import CheckpointManager
from .population_selection import PopulationSelectionEngine, CandidateRecord, CandidateStatus, CapabilityVector, ModelFamily
from .reputation import ReputationStore, ModelReputation, ContributionLedger
from .verification import RecursiveVerifier, MetaVerifier, VerificationResult
from .calibration import CalibrationTracker, UncertaintyEstimator, UncertaintyProfile
from .correlated_failures import CorrelatedFailureDetector
from .information_gain import InformationGainEstimator

from ..utils.paths import get_checkpoints_root


@dataclass
class ModelTrainingContext:
    """Context for a single model's training."""
    model_id: str
    trainer: Optional[BaseTrainer] = None
    budget: Optional[TrainingBudget] = None
    quality_profile: Optional[QualityProfile] = None
    best_run_metrics: Optional[RunMetrics] = None
    knowledge_sources: List[str] = field(default_factory=list)
    hard_examples: List[str] = field(default_factory=list)
    lineage: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    candidate_record: Optional[CandidateRecord] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "budget": self.budget.to_dict() if self.budget else None,
            "quality_profile": self.quality_profile.to_dict() if self.quality_profile else None,
            "best_run_metrics": self.best_run_metrics.to_dict() if self.best_run_metrics else None,
            "knowledge_sources": self.knowledge_sources,
            "hard_examples": self.hard_examples,
            "lineage": self.lineage,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
        }


class TrainingController:
    """
    Centralized controller for population training.
    
    Coordinates:
    - Budget allocation
    - Quality monitoring
    - Multi-run management
    - Hard-example replay
    - Distillation
    - Curriculum updates
    - Population evolution
    - Quality-first training loop (train → evaluate → diagnose → compare → retain/restart)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.budget_allocator = AdaptiveBudgetAllocator(config.get("adaptive_budget", {}))
        self.quality_lock = QualityLock(config.get("quality_lock", {}))
        self.scheduler = PopulationScheduler(config.get("scheduler", {}))
        self.evolution = PopulationEvolution(config)
        self.hard_example_miner = HardExampleMiner(config)
        self.curriculum_updater = CurriculumUpdater(config)
        self.selection_engine = PopulationSelectionEngine(config.get("population_selection", {}))
        self.reputation_store = ReputationStore(config.get("reputation", {}).get("storage_path"))
        self.recursive_verifier = RecursiveVerifier(config)
        self.meta_verifier = MetaVerifier(config)
        self.calibration_tracker = CalibrationTracker(config)
        self.uncertainty_estimator = UncertaintyEstimator(config)
        self.correlated_failure_detector = CorrelatedFailureDetector(config)
        self.information_gain_estimator = InformationGainEstimator(config)
        self.model_contexts: Dict[str, ModelTrainingContext] = {}
        self._lock = threading.Lock()
        
        self.model_contexts: Dict[str, ModelTrainingContext] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
    
    def register_model(self, model_id: str, profile: Dict[str, Any]) -> ModelTrainingContext:
        """Register a model for training."""
        context = ModelTrainingContext(model_id=model_id, metadata=profile)
        with self._lock:
            self.model_contexts[model_id] = context
        
        # Enqueue for training
        record = ModelTrainingRecord(
            priority=profile.get("priority", 1.0),
            model_id=model_id,
            deadline_timestamp=profile.get("deadline_timestamp"),
            hardware_requirement=profile.get("hardware_requirement", {}),
        )
        self.scheduler.enqueue(record)
        return context
    
    def allocate_budgets(self, model_profiles: List[Dict[str, Any]]) -> BudgetAllocation:
        """Allocate training budgets for a batch of models."""
        return self.budget_allocator.allocate_initial_budget(model_profiles)
    
    @staticmethod
    def _compute_capability_scores(val_loss: float, train_loss: float) -> Dict[str, float]:
        """Map validation loss to capability scores.

        This is a deterministic, loss-based evaluation used when no external
        benchmark is available. It is NOT a production benchmark score.
        """
        if not math.isfinite(val_loss) or not math.isfinite(train_loss):
            return {
                "reasoning_score": 0.0,
                "knowledge_score": 0.0,
                "domain_score": 0.0,
                "verification_score": 0.0,
                "stability_score": 0.0,
                "overfit_score": 0.0,
            }
        base = max(0.0, min(1.0, (12.0 - val_loss) / 8.0))
        overfit_gap = train_loss - val_loss
        overfit_penalty = max(0.0, overfit_gap) * 0.5
        return {
            "reasoning_score": max(0.0, base - 0.05),
            "knowledge_score": max(0.0, base * 0.95),
            "domain_score": max(0.0, base * 0.9),
            "verification_score": max(0.0, base * 0.85),
            "stability_score": max(0.0, base - 0.1),
            "overfit_score": max(0.0, min(1.0, 1.0 - overfit_penalty)),
        }

    def run_multi_run_training(self, model_id: str, model_class, train_dataloader, eval_dataloader, num_runs: int = 3, trainer_class: type = None) -> MultiRunResult:
        """Execute multi-run training for a model.

        Args:
            model_id: Identifier for the current model.
            model_class: Callable that accepts ``model_id`` and returns a fresh
                ``nn.Module`` instance.
        """
        context = self.model_contexts.get(model_id)
        if context is None:
            raise ValueError(f"Model {model_id} not registered")
        
        if context.budget is None:
            raise ValueError(f"Model {model_id} has no allocated budget")
        
        output_dir = get_checkpoints_root() / "multi_run" / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        runs: List[RunMetrics] = []
        start_time = time.time()

        score_weights = {
            "val_loss": 0.3,
            "reasoning_score": 0.2,
            "knowledge_score": 0.15,
            "domain_score": 0.1,
            "verification_score": 0.1,
            "stability_score": 0.1,
            "overfit_score": 0.05,
        }
        trainer_cls = trainer_class or BaseTrainer
        per_model_config = dict(self.config)
        if context.metadata:
            per_model_training = context.metadata.get("training", {})
            if per_model_training:
                base_training = per_model_config.get("training", {})
                base_training = dict(base_training)
                base_training.update(per_model_training)
                per_model_config["training"] = base_training

        for run_id in range(num_runs):
            run_start = time.time()
            seed = per_model_config.get("training", {}).get("seed", 42) + run_id
            torch.manual_seed(seed)

            model = model_class(model_id)
            run_dir = output_dir / f"run_{run_id:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            trainer = trainer_cls(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                config=per_model_config,
                output_dir=str(run_dir),
            )
            trainer.setup_optimizer()
            trainer.train()
            tokens_seen = getattr(trainer.state, "tokens_seen", 0)

            eval_metrics = trainer.evaluate()
            train_loss = eval_metrics.get("eval_loss", float("inf"))
            val_loss = eval_metrics.get("eval_loss", train_loss)

            metrics = RunMetrics(
                run_id=run_id,
                train_loss=train_loss,
                val_loss=val_loss,
                checkpoint_path=str(run_dir / "final_model.safetensors"),
                duration_seconds=time.time() - run_start,
                tokens_seen=tokens_seen,
            )
            caps = self._compute_capability_scores(val_loss, train_loss)
            metrics.reasoning_score = caps["reasoning_score"]
            metrics.knowledge_score = caps["knowledge_score"]
            metrics.domain_score = caps["domain_score"]
            metrics.verification_score = caps["verification_score"]
            metrics.stability_score = caps["stability_score"]
            metrics.overfit_score = caps["overfit_score"]
            metrics.compute_composite(score_weights)
            runs.append(metrics)

        best_run = max(runs, key=lambda r: r.composite_score)
        result = MultiRunResult(
            best_run_id=best_run.run_id,
            best_run_metrics=best_run,
            all_runs=runs,
            total_runs=num_runs,
            total_duration_seconds=time.time() - start_time,
        )
        context.best_run_metrics = best_run
        return result
    
    def evaluate_model_quality(self, model_id: str, baseline_scores: Dict[str, float], current_scores: Dict[str, float]) -> QualityProfile:
        """Evaluate model quality against baseline."""
        profile = self.quality_lock.create_profile(model_id, baseline_scores, current_scores)
        context = self.model_contexts.get(model_id)
        if context is not None:
            context.quality_profile = profile
        return profile
    
    def validate_quality(self, profile: QualityProfile) -> Dict[str, Any]:
        """Validate quality retention."""
        return self.quality_lock.validate_optimization(profile)
    
    def check_catastrophic_forgetting(self, before_scores: Dict[str, float], after_scores: Dict[str, float]) -> Dict[str, Any]:
        """Check for catastrophic forgetting."""
        return self.quality_lock.check_catastrophic_forgetting(before_scores, after_scores)
    
    def should_stop_training(self, model_id: str, budget: TrainingBudget) -> Optional[StopReason]:
        """Determine if a model should stop training."""
        return budget.should_stop(
            marginal_gain_threshold=self.config.get("marginal_gain_threshold", 0.001),
            plateau_patience=self.config.get("plateau_patience", 3),
        )
    
    def should_restart(self, model_id: str, budget: TrainingBudget) -> bool:
        """Determine if a model should get a fresh restart."""
        threshold = self.config.get("adaptive_budget", {}).get("restart_value_threshold", 0.05)
        return self.budget_allocator.should_restart(budget, restart_value_threshold=threshold)
    
    def update_budget_progress(self, model_id: str, tokens_seen: int, wall_time: float, current_quality: float, marginal_gain: float) -> Optional[TrainingBudget]:
        """Update budget progress for a model."""
        context = self.model_contexts.get(model_id)
        if context is None or context.budget is None:
            return None
        
        budget = context.budget
        budget.tokens_seen += tokens_seen
        budget.wall_time_seconds += wall_time
        budget.current_quality = current_quality
        budget.marginal_gain = marginal_gain
        
        # Reallocate if needed
        allocation = BudgetAllocation(
            model_budgets={model_id: budget},
            total_wall_time=budget.wall_time_seconds,
            total_tokens=budget.tokens_seen,
            total_compute=budget.compute_used,
        )
        self.budget_allocator.reallocate_budget(allocation, {model_id: current_quality}, {model_id: marginal_gain})
        
        return budget
    
    def mine_hard_examples(self, model_id: str, task_id: str, prompt: str, candidates: Dict[str, Any], verifications: Dict[str, Any]) -> Optional[Any]:
        """Mine hard examples from model failures."""
        example = self.hard_example_miner.mine_from_population(task_id, prompt, candidates, verifications)
        if example is not None:
            context = self.model_contexts.get(model_id)
            if context is not None:
                context.hard_examples.append(example.example_id)
        return example
    
    def update_curriculum(self, mixture: Any, hard_examples: List[Any], failure_stats: Dict[str, Any]) -> Any:
        """Update curriculum based on hard examples."""
        return self.curriculum_updater.update_mixture(mixture, hard_examples, failure_stats)
    
    def evolve_population(self, model_ids: List[str], hard_examples: List[Any]) -> List[Any]:
        """Evolve population based on fitness and hard examples."""
        return self.evolution.evolve_population(model_ids, hard_examples)
    
    def get_model_context(self, model_id: str) -> Optional[ModelTrainingContext]:
        """Get training context for a model."""
        return self.model_contexts.get(model_id)
    
    def get_population_stats(self) -> Dict[str, Any]:
        """Get population training statistics."""
        with self._lock:
            total = len(self.model_contexts)
            complete = sum(1 for c in self.model_contexts.values() if c.best_run_metrics is not None)
            return {
                "total_models": total,
                "complete": complete,
                "in_progress": total - complete,
                "queue_stats": self.scheduler.get_queue_stats(),
            }

    # ------------------------------------------------------------------
    # Quality-first population training loop
    # ------------------------------------------------------------------

    def run_population_training_loop(
        self,
        model_profiles: List[Dict[str, Any]],
        model_class: type,
        build_dataloaders: Callable[[str, Dict[str, Any]], tuple],
        num_runs: int = 3,
        convergence_window: int = 3,
        convergence_threshold: float = 0.001,
        trainer_class: type = None,
        post_run_injector: Optional[Callable[[str, RunMetrics], Optional[RunMetrics]]] = None,
    ) -> Dict[str, Any]:
        """
        Run the quality-first population training loop.

        Policy: maximize final capability. Do not reduce dataset quality,
        verification, training duration, diversity, deliberation, evaluation
        coverage, or retry opportunities solely to reduce compute.

        Loop per candidate:
          1. Train candidate (multi-run with fresh seeds)
          2. Evaluate (perplexity + capability benchmarks)
          3. Diagnose failure profile
          4. Compare against population
          5. Retain strong candidates
          6. Restart weak candidates from scratch when appropriate
          7. Repeat until quality converges or budget exhausted

        Args:
            model_profiles: list of dicts with keys:
                model_id, priority, quality_target, max_steps, ...
            model_class: Callable that accepts ``model_id`` and returns a fresh
                ``nn.Module`` instance.
            build_dataloaders: callable(model_id, profile) -> (train_dl, eval_dl)
            num_runs: fresh-seed runs per candidate before selection
            convergence_window: number of recent quality deltas to average
            convergence_threshold: average delta below which we declare convergence
            trainer_class: optional concrete ``BaseTrainer`` subclass to use.
            post_run_injector: optional hook ``(model_id, best_metrics) -> RunMetrics | None``
                used to inject labeled ``controller_behavior_test`` metrics after training
                but before evaluation, diagnosis, and decision.

        Returns:
            Population training summary dict.
        """
        # 1. Register and allocate budgets
        for profile in model_profiles:
            self.register_model(profile["model_id"], profile)
        allocation = self.allocate_budgets(model_profiles)

        # Assign budgets to contexts
        with self._lock:
            for model_id, budget in allocation.model_budgets.items():
                if model_id in self.model_contexts:
                    self.model_contexts[model_id].budget = budget

        generation = 1
        population_records: Dict[str, CandidateRecord] = {}
        summary = {
            "generation": generation,
            "trained": [],
            "retained": [],
            "restarted": [],
            "rejected": [],
            "diagnostics": {},
            "decision_traces": [],
        }

        max_generations = self.config.get("max_generations", 10)
        quality_history: Dict[str, List[float]] = {p["model_id"]: [] for p in model_profiles}

        for gen in range(1, max_generations + 1):
            if self._stop_event.is_set():
                break

            summary["generation"] = gen
            trained_this_gen = []
            restarted_this_gen = []
            retained_this_gen = []
            dequeue_count = 0

            # Dequeue and process all queued models
            while True:
                record = self.scheduler.dequeue()
                if record is None:
                    break
                dequeue_count += 1
                print(f"[QUEUE_DEBUG] gen={gen} dequeue_count={dequeue_count} model={record.model_id} status={record.status} attempts={record.attempts}")

                model_id = record.model_id
                context = self.model_contexts.get(model_id)
                if context is None or context.budget is None:
                    self.scheduler.mark_failed(model_id, retry=False)
                    continue

                budget = context.budget
                stop_reason = self.should_stop_training(model_id, budget)
                if stop_reason is not None:
                    self.scheduler.mark_complete(model_id, quality=budget.current_quality)
                    continue

                # 2. Train candidate
                train_dl, eval_dl = build_dataloaders(model_id, context.metadata)
                try:
                    multi_run_result = self.run_multi_run_training(
                        model_id=model_id,
                        model_class=model_class,
                        train_dataloader=train_dl,
                        eval_dataloader=eval_dl,
                        num_runs=num_runs,
                        trainer_class=trainer_class,
                    )
                except Exception as exc:
                    print(f"[training_controller] Training failed for {model_id}: {exc}")
                    self.scheduler.mark_failed(model_id, retry=True)
                    continue

                best_metrics = multi_run_result.best_run_metrics
                context.best_run_metrics = best_metrics

                if post_run_injector is not None:
                    injected = post_run_injector(model_id, best_metrics)
                    if injected is not None:
                        best_metrics = injected
                        context.best_run_metrics = injected
                        multi_run_result.best_run_metrics = injected
                        multi_run_result.all_runs = [
                            injected if r.run_id == injected.run_id else r
                            for r in multi_run_result.all_runs
                        ]

                # Update budget with actual training cost
                budget.wall_time_seconds += best_metrics.duration_seconds * num_runs
                total_run_tokens = sum(getattr(r, "tokens_seen", 0) or 0 for r in multi_run_result.all_runs)
                budget.tokens_seen += total_run_tokens

                # 3. Evaluate
                eval_scores = {
                    "val_loss": best_metrics.val_loss,
                    "reasoning_score": best_metrics.reasoning_score,
                    "knowledge_score": best_metrics.knowledge_score,
                    "domain_score": best_metrics.domain_score,
                    "verification_score": best_metrics.verification_score,
                    "stability_score": best_metrics.stability_score,
                    "overfit_score": best_metrics.overfit_score,
                }
                current_quality = best_metrics.composite_score
                prev_quality = budget.current_quality
                budget.current_quality = current_quality
                budget.marginal_gain = max(0.0, current_quality - prev_quality)

                # Advance budget phase after first completed run
                if budget.phase == BudgetPhase.WARMUP.value:
                    budget.phase = BudgetPhase.ADAPTIVE.value

                baseline_scores = {k: 0.0 for k in eval_scores}
                quality_profile = self.evaluate_model_quality(model_id, baseline_scores, eval_scores)
                context.quality_profile = quality_profile

                # 4. Diagnose
                diagnostics = self._diagnose_candidate(model_id, best_metrics, quality_profile)
                context.diagnostics = diagnostics

                # 5. Compare against population
                capability_vector = CapabilityVector(
                    general_knowledge=eval_scores.get("knowledge_score", 0.0),
                    language=eval_scores.get("domain_score", 0.0),
                    mathematics=eval_scores.get("reasoning_score", 0.0),
                    programming=eval_scores.get("verification_score", 0.0),
                    computer_science=eval_scores.get("stability_score", 0.0),
                    science=eval_scores.get("overfit_score", 0.0),
                    reasoning=eval_scores.get("reasoning_score", 0.0),
                    verification=eval_scores.get("verification_score", 0.0),
                    planning=eval_scores.get("domain_score", 0.0),
                    tool_use=eval_scores.get("stability_score", 0.0),
                    memory=eval_scores.get("knowledge_score", 0.0),
                    collaboration=eval_scores.get("domain_score", 0.0),
                    computer_vision=eval_scores.get("verification_score", 0.0),
                    video_understanding=eval_scores.get("stability_score", 0.0),
                    computer_use=eval_scores.get("reasoning_score", 0.0),
                )
                candidate = CandidateRecord(
                    model_id=model_id,
                    overall_quality=current_quality,
                    capability_vector=capability_vector,
                    checkpoint_path=best_metrics.checkpoint_path,
                    training_run=multi_run_result.to_dict(),
                    failure_categories=diagnostics.get("failure_categories", []),
                )
                candidate = self.selection_engine.score_candidate(candidate)
                self.selection_engine.compute_diversity_scores([candidate])
                population_records[model_id] = candidate
                context.candidate_record = candidate

                # 6. Retention / restart decision
                stop_reason_after = self.should_stop_training(model_id, budget)
                if stop_reason_after is not None:
                    self.scheduler.mark_complete(model_id, quality=current_quality)
                    retained_this_gen.append(model_id)
                    trained_this_gen.append(model_id)
                    trace = self._build_decision_trace(
                        model_id, best_metrics.run_id, budget, candidate,
                        decision="STOP",
                        reason=f"stop_reason={stop_reason_after.value}",
                        convergence_state="stopped",
                    )
                    summary["decision_traces"].append(trace)
                    print(f"[DECISION] {model_id}: STOP reason={stop_reason_after.value} composite={candidate.composite_score:.4f}")
                    continue

                # Decide restart
                restart = self._should_restart_candidate(model_id, budget, record, diagnostics, candidate)
                print(f"[DEBUG] {model_id}: restart={restart}, candidate_composite={candidate.composite_score}, quality_floor={self.selection_engine.quality_floor}")
                if restart:
                    record.attempts += 1
                    budget.tokens_seen = 0
                    budget.wall_time_seconds = 0.0
                    budget.current_quality = 0.0
                    budget.marginal_gain = 0.0
                    budget.phase = BudgetPhase.WARMUP.value
                    budget.quality_plateau_count = 0
                    record.status = ModelTrainingStatus.QUEUED.value
                    record.started_at = None
                    self.scheduler._queue.put(record)
                    restarted_this_gen.append(model_id)
                    trained_this_gen.append(model_id)
                    trace = self._build_decision_trace(
                        model_id, best_metrics.run_id, budget, candidate,
                        decision="RESTART",
                        reason="recoverable failure detected: " + ", ".join(diagnostics.get("failure_categories", [])),
                        convergence_state="restarting",
                    )
                    summary["decision_traces"].append(trace)
                    print(f"[DECISION] {model_id}: RESTART reason=recoverable_failure composite={candidate.composite_score:.4f}")
                    continue

                # Retain if quality is good, otherwise reject
                if candidate.composite_score >= self.selection_engine.quality_floor:
                    self.scheduler.mark_complete(model_id, quality=current_quality)
                    retained_this_gen.append(model_id)
                    trace = self._build_decision_trace(
                        model_id, best_metrics.run_id, budget, candidate,
                        decision="RETAIN",
                        reason=f"composite_score={candidate.composite_score:.4f} >= quality_floor={self.selection_engine.quality_floor}",
                        convergence_state="retained",
                    )
                    summary["decision_traces"].append(trace)
                    print(f"[DECISION] {model_id}: RETAIN composite={candidate.composite_score:.4f}")
                else:
                    print(f"[DEBUG] {model_id}: calling mark_failed(retry=False)")
                    self.scheduler.mark_failed(model_id, retry=False)
                    summary["rejected"].append({
                        "model_id": model_id,
                        "composite_score": candidate.composite_score,
                        "diagnostics": diagnostics,
                    })
                    trace = self._build_decision_trace(
                        model_id, best_metrics.run_id, budget, candidate,
                        decision="REJECT",
                        reason=f"composite_score={candidate.composite_score:.4f} < quality_floor={self.selection_engine.quality_floor}",
                        convergence_state="rejected",
                    )
                    summary["decision_traces"].append(trace)
                    print(f"[DECISION] {model_id}: REJECT reason=below_quality_floor composite={candidate.composite_score:.4f}")
                trained_this_gen.append(model_id)
                quality_history[model_id].append(current_quality)

            # Population-wide comparison and selection
            all_candidates = list(population_records.values())
            if all_candidates:
                retained, discarded = self._run_population_selection(all_candidates)
                summary["retained"].extend([r.model_id for r in retained])
                for c in discarded:
                    if c.model_id not in summary["rejected"]:
                        summary["rejected"].append({
                            "model_id": c.model_id,
                            "composite_score": c.composite_score,
                            "diagnostics": self.model_contexts.get(c.model_id, ModelTrainingContext(c.model_id)).diagnostics,
                        })

            # Hard-example mining and curriculum update
            if trained_this_gen:
                hard_examples = []
                for mid in trained_this_gen:
                    ctx = self.model_contexts.get(mid)
                    if ctx and ctx.candidate_record:
                        for fc in ctx.candidate_record.failure_categories:
                            hard_examples.append({"model_id": mid, "failure_category": fc})
                            failure_log = FailureLog(
                                failure_id=f"{mid}_{fc}_{int(time.time()*1000)}",
                                model_id=mid,
                                task_id=mid,
                                category=fc,
                                description=f"Training failure: {fc}",
                                metadata={"composite_score": ctx.candidate_record.composite_score},
                            )
                            self.hard_example_miner.failure_logs.append(failure_log)
                            self.correlated_failure_detector.record_failure(
                                model_id=mid,
                                task_id=mid,
                                failure_category=fc,
                                description=f"Training failure: {fc}",
                                confidence=ctx.candidate_record.composite_score,
                                metadata={"composite_score": ctx.candidate_record.composite_score},
                            )
                if hard_examples:
                    self.evolution.evolve_population(trained_this_gen, [])

            # Update reputation and contribution tracking
            for mid in trained_this_gen:
                ctx = self.model_contexts.get(mid)
                if ctx and ctx.candidate_record:
                    reputation = self.reputation_store.get_or_create(
                        mid, ctx.candidate_record.model_family
                    )
                    is_success = ctx.candidate_record.status in (
                        CandidateStatus.RETAINED.value,
                        CandidateStatus.PENDING.value,
                    )
                    reputation.record_global_attempt(
                        success=is_success,
                        confidence=ctx.candidate_record.composite_score,
                        actual_correct=is_success,
                        is_hallucination="HALLUCINATION" in ctx.candidate_record.failure_categories,
                        is_error=bool(ctx.candidate_record.failure_categories),
                    )
                    reputation.contribution_ledger.record_participation(contributed=is_success)
                    self.reputation_store.update_reputation(reputation)
                    ctx.candidate_record.reputation_tier = reputation.tier
                    ctx.candidate_record.contribution_score = reputation.contribution_ledger.contribution_rate

            summary["trained"].extend(trained_this_gen)
            summary["restarted"].extend(restarted_this_gen)

            # Convergence check
            if self._check_convergence(quality_history, convergence_window, convergence_threshold):
                print(f"[training_controller] Population converged at generation {gen}")
                break

            # Budget reallocation
            quality_updates = {mid: quality_history[mid][-1] if quality_history[mid] else 0.0 for mid in quality_history}
            marginal_gains = {}
            for mid, ctx in self.model_contexts.items():
                if ctx.budget is not None:
                    marginal_gains[mid] = ctx.budget.marginal_gain
            allocation = BudgetAllocation(
                model_budgets={mid: ctx.budget for mid, ctx in self.model_contexts.items() if ctx.budget is not None},
                total_wall_time=sum(ctx.budget.wall_time_seconds for ctx in self.model_contexts.values() if ctx.budget),
                total_tokens=sum(ctx.budget.tokens_seen for ctx in self.model_contexts.values() if ctx.budget),
                total_compute=sum(ctx.budget.compute_used for ctx in self.model_contexts.values() if ctx.budget),
            )
            self.budget_allocator.reallocate_budget(allocation, quality_updates, marginal_gains)
            with self._lock:
                for mid, budget in allocation.model_budgets.items():
                    if mid in self.model_contexts:
                        self.model_contexts[mid].budget = budget

        summary["quality_history"] = quality_history
        summary["population_records"] = {mid: cr.to_dict() for mid, cr in population_records.items()}
        return summary

    def _train_single_candidate(
        self,
        model_id: str,
        model_class: type,
        train_dataloader,
        eval_dataloader,
        num_runs: int = 3,
    ) -> MultiRunResult:
        """Train a single candidate with multiple fresh seeds."""
        return self.run_multi_run_training(
            model_id=model_id,
            model_class=model_class,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            num_runs=num_runs,
        )

    def _evaluate_candidate(self, model_id: str, metrics: RunMetrics) -> Dict[str, float]:
        """Evaluate a candidate and return capability scores."""
        return {
            "val_loss": metrics.val_loss,
            "reasoning_score": metrics.reasoning_score,
            "knowledge_score": metrics.knowledge_score,
            "domain_score": metrics.domain_score,
            "verification_score": metrics.verification_score,
            "stability_score": metrics.stability_score,
            "overfit_score": metrics.overfit_score,
        }

    def _diagnose_candidate(self, model_id: str, metrics: RunMetrics, quality_profile: QualityProfile) -> Dict[str, Any]:
        """Diagnose failure profile for a candidate."""
        diagnostics: Dict[str, Any] = {
            "model_id": model_id,
            "val_loss": metrics.val_loss,
            "train_loss": metrics.train_loss,
            "composite_score": metrics.composite_score,
            "quality_tier": quality_profile.get_quality_tier().value if quality_profile else "unknown",
            "failure_categories": [],
            "weak_capabilities": [],
            "overfit_gap": metrics.train_loss - metrics.val_loss,
        }

        # Detect weak capabilities
        weak_threshold = 0.4
        capability_map = {
            "reasoning": metrics.reasoning_score,
            "knowledge": metrics.knowledge_score,
            "domain": metrics.domain_score,
            "verification": metrics.verification_score,
            "stability": metrics.stability_score,
        }
        diagnostics["weak_capabilities"] = [name for name, score in capability_map.items() if score < weak_threshold]

        # Classify failures
        if metrics.val_loss > 2.0:
            diagnostics["failure_categories"].append("HIGH_VAL_LOSS")
        if metrics.train_loss - metrics.val_loss > 0.5:
            diagnostics["failure_categories"].append("OVERFIT")
        elif metrics.val_loss - metrics.train_loss > 0.5:
            diagnostics["failure_categories"].append("UNDERFIT")
        if metrics.verification_score < 0.3:
            diagnostics["failure_categories"].append("VERIFICATION_WEAK")
        if metrics.reasoning_score < 0.3:
            diagnostics["failure_categories"].append("REASONING_WEAK")
        if metrics.stability_score < 0.3:
            diagnostics["failure_categories"].append("TRAINING_UNSTABLE")

        return diagnostics

    def _should_restart_candidate(
        self,
        model_id: str,
        budget: TrainingBudget,
        record: Any,
        diagnostics: Dict[str, Any],
        candidate: CandidateRecord,
    ) -> bool:
        """Decide whether a candidate should restart from scratch."""
        if not self.budget_allocator.should_restart(budget):
            return False

        if candidate.composite_score < 0.2:
            return False

        recoverable_failures = {"OVERFIT", "UNDERFIT", "TRAINING_UNSTABLE", "HIGH_VAL_LOSS"}
        if any(fc in recoverable_failures for fc in diagnostics.get("failure_categories", [])):
            return True

        return False

    def _run_population_selection(self, candidates: List[CandidateRecord]) -> tuple:
        """Run population selection and return (retained, discarded)."""
        return self.selection_engine.select(candidates)

    def _check_convergence(self, quality_history: Dict[str, List[float]], window: int, threshold: float) -> bool:
        """Check if population quality has converged."""
        recent_deltas = []
        for model_id, history in quality_history.items():
            if len(history) < 2:
                continue
            recent = history[-window:]
            if len(recent) >= 2:
                deltas = [abs(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
                if deltas:
                    recent_deltas.append(sum(deltas) / len(deltas))
        if not recent_deltas:
            return False
        avg_delta = sum(recent_deltas) / len(recent_deltas)
        return avg_delta < threshold

    def _build_decision_trace(self, model_id: str, run_id: int, budget: TrainingBudget, candidate: CandidateRecord, decision: str, reason: str, convergence_state: str = "unknown") -> Dict[str, Any]:
        """Build an explicit decision trace entry for reporting."""
        return {
            "model_id": model_id,
            "run_id": run_id,
            "quality": budget.current_quality,
            "quality_delta": budget.marginal_gain,
            "validation": candidate.composite_score if candidate else None,
            "tokens_seen": budget.tokens_seen,
            "budget": budget.to_dict(),
            "remaining_budget": {
                "remaining_wall_time": budget.remaining_wall_time(),
                "remaining_tokens": budget.remaining_tokens(),
                "remaining_compute": budget.remaining_compute(),
            },
            "marginal_gain": budget.marginal_gain,
            "convergence_state": convergence_state,
            "restart_count": getattr(candidate, "restart_count", 0) if candidate else 0,
            "decision": decision,
            "decision_reason": reason,
        }
    
    def shutdown(self) -> None:
        """Shutdown controller."""
        self._stop_event.set()
