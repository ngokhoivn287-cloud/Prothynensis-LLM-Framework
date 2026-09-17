"""Tests for adaptive population training: quality lock, adaptive budget, scheduler, controller, knowledge reuse, lineage."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.training.quality_lock import QualityLock, QualityProfile, CapabilityQuality, QualityTier
from src.training.adaptive_budget import AdaptiveBudgetAllocator, BudgetAllocation, TrainingBudget, BudgetPhase, StopReason
from src.training.population_scheduler import PopulationScheduler, ModelTrainingRecord, ModelTrainingStatus
from src.training.training_controller import TrainingController, ModelTrainingContext
from src.training.population_selection import CandidateRecord
from src.training.hard_examples import FailureLog
from src.training.knowledge_reuse import KnowledgeStore, KnowledgeReuseManager, KnowledgeEntry, ReusePolicyConfig, KnowledgeType, ReusePolicy
from src.training.lineage import LineageTracker, ModelLineage, LineageEvent, LineageEventType


class TestQualityLock:
    def test_passes_global(self):
        lock = QualityLock({"global_threshold": 0.96})
        baseline = {"reasoning": 0.9, "knowledge": 0.85}
        current = {"reasoning": 0.92, "knowledge": 0.87}
        profile = lock.create_profile("m1", baseline, current)
        assert profile.passes_all() is True
    
    def test_fails_per_capability(self):
        lock = QualityLock({"per_capability_threshold": 0.965})
        baseline = {"reasoning": 0.9, "knowledge": 0.9}
        current = {"reasoning": 0.9, "knowledge": 0.85}
        profile = lock.create_profile("m1", baseline, current)
        assert profile.passes_all() is False
        assert "knowledge" in profile.get_failing_capabilities()
    
    def test_rejects_below_threshold(self):
        lock = QualityLock({"global_threshold": 0.965})
        baseline = {"reasoning": 1.0}
        current = {"reasoning": 0.9}
        profile = lock.create_profile("m1", baseline, current)
        result = lock.validate_optimization(profile)
        assert result["tier"] == QualityTier.REJECTED.value
        assert result["passes"] is False
    
    def test_excellent_tier(self):
        lock = QualityLock({"global_threshold": 0.965, "global_target": 0.98})
        baseline = {"reasoning": 1.0}
        current = {"reasoning": 0.995}
        profile = lock.create_profile("m1", baseline, current)
        result = lock.validate_optimization(profile)
        assert result["tier"] == QualityTier.EXCELLENT.value
    
    def test_check_catastrophic_forgetting(self):
        lock = QualityLock()
        before = {"math": 0.9, "code": 0.9, "reasoning": 0.9}
        after = {"math": 0.95, "code": 0.95, "reasoning": 0.85}
        result = lock.check_catastrophic_forgetting(before, after)
        assert result["catastrophic_forgetting"] is True
        assert any(d["capability"] == "reasoning" for d in result["degraded_capabilities"])


class TestAdaptiveBudget:
    def test_allocate_initial(self):
        allocator = AdaptiveBudgetAllocator({"default_token_budget": 1_000_000, "default_wall_time_budget": 86400})
        profiles = [
            {"model_id": "m1", "priority": 1.0, "quality_target": 0.9},
            {"model_id": "m2", "priority": 0.5, "quality_target": 0.85},
        ]
        allocation = allocator.allocate_initial_budget(profiles)
        assert "m1" in allocation.model_budgets
        assert "m2" in allocation.model_budgets
        assert allocation.model_budgets["m1"].max_tokens >= allocation.model_budgets["m2"].max_tokens
    
    def test_should_stop_quality_target(self):
        allocator = AdaptiveBudgetAllocator({})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=1000, max_tokens=1000, max_compute_flops=1.0, quality_target=0.9)
        budget.current_quality = 0.95
        reason = budget.should_stop()
        assert reason == StopReason.QUALITY_TARGET_REACHED
    
    def test_should_stop_deadline(self):
        import time as _time
        allocator = AdaptiveBudgetAllocator({})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=10, max_tokens=100000, max_compute_flops=1.0, quality_target=0.9, deadline_timestamp=_time.time() - 10)
        reason = budget.should_stop()
        assert reason == StopReason.DEADLINE_REACHED
    
    def test_rtv_computation(self):
        allocator = AdaptiveBudgetAllocator({})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=1000, max_tokens=1000, max_compute_flops=1.0, quality_target=0.9)
        budget.current_quality = 0.5
        rtv = allocator.compute_rtv(budget)
        assert rtv > 0
    
    def test_prioritize_models(self):
        allocator = AdaptiveBudgetAllocator({})
        b1 = TrainingBudget(model_id="m1", max_wall_time_seconds=1000, max_tokens=1000, max_compute_flops=1.0, quality_target=0.9)
        b1.current_quality = 0.5
        b2 = TrainingBudget(model_id="m2", max_wall_time_seconds=1000, max_tokens=1000, max_compute_flops=1.0, quality_target=0.9)
        b2.current_quality = 0.8
        allocation = BudgetAllocation(model_budgets={"m1": b1, "m2": b2}, total_wall_time=2000, total_tokens=2000, total_compute=2.0)
        order = allocator.prioritize_models(allocation)
        # m1 has lower quality, so more remaining quality to gain -> higher RTV
        assert order[0] == "m1"

    def test_restart_token_fraction_default(self):
        allocator = AdaptiveBudgetAllocator({"default_token_budget": 500_000, "default_wall_time_budget": 1000})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=1000, max_tokens=500_000, max_compute_flops=1.0, quality_target=0.9)
        budget.current_quality = 0.5
        budget.phase = BudgetPhase.ADAPTIVE.value
        budget.tokens_seen = 249_999
        assert allocator.should_restart(budget) is False
        budget.tokens_seen = 250_000
        assert allocator.should_restart(budget) is True

    def test_restart_token_fraction_override(self):
        allocator = AdaptiveBudgetAllocator({"default_token_budget": 500_000, "default_wall_time_budget": 1000, "restart_token_fraction": 0.01})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=1000, max_tokens=500_000, max_compute_flops=1.0, quality_target=0.9)
        budget.current_quality = 0.5
        budget.phase = BudgetPhase.ADAPTIVE.value
        budget.tokens_seen = 4_999
        assert allocator.should_restart(budget) is False
        budget.tokens_seen = 5_000
        assert allocator.should_restart(budget) is True

    def test_restart_gate_allows_small_stress_budget(self):
        allocator = AdaptiveBudgetAllocator({"default_token_budget": 500_000, "default_wall_time_budget": 604800, "restart_token_fraction": 0.01})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=604800, max_tokens=500_000, max_compute_flops=1.0, quality_target=0.9)
        budget.current_quality = 0.5
        budget.phase = BudgetPhase.ADAPTIVE.value
        budget.tokens_seen = 12_800
        assert allocator.should_restart(budget) is True


class TestPopulationScheduler:
    def test_enqueue_and_dequeue(self):
        scheduler = PopulationScheduler({})
        record = ModelTrainingRecord(priority=1.0, model_id="m1")
        scheduler.enqueue(record)
        dequeued = scheduler.dequeue()
        assert dequeued is not None
        assert dequeued.model_id == "m1"
        assert dequeued.status == ModelTrainingStatus.TRAINING.value
    
    def test_mark_complete(self):
        scheduler = PopulationScheduler({})
        record = ModelTrainingRecord(priority=1.0, model_id="m1")
        scheduler.enqueue(record)
        scheduler.dequeue()
        scheduler.mark_complete("m1", quality=0.9)
        assert scheduler.get_status("m1") == ModelTrainingStatus.COMPLETE.value
    
    def test_mark_failed_retry(self):
        scheduler = PopulationScheduler({})
        record = ModelTrainingRecord(priority=1.0, model_id="m1", max_attempts=2)
        scheduler.enqueue(record)
        scheduler.dequeue()
        scheduler.mark_failed("m1", retry=True)
        assert scheduler.get_status("m1") == ModelTrainingStatus.QUEUED.value
    
    def test_queue_stats(self):
        scheduler = PopulationScheduler({})
        for i in range(3):
            record = ModelTrainingRecord(priority=1.0, model_id=f"m{i}")
            scheduler.enqueue(record)
        stats = scheduler.get_queue_stats()
        assert stats["total"] == 3
    
    def test_save_and_load_state(self, tmp_path):
        scheduler = PopulationScheduler({})
        record = ModelTrainingRecord(priority=1.0, model_id="m1")
        scheduler.enqueue(record)
        path = str(tmp_path / "scheduler_state.json")
        scheduler.save_state(path)
        
        scheduler2 = PopulationScheduler({})
        scheduler2.load_state(path)
        assert scheduler2.get_status("m1") == ModelTrainingStatus.QUEUED.value


class TestTrainingController:
    def test_register_model(self):
        controller = TrainingController({})
        context = controller.register_model("m1", {"priority": 1.0})
        assert context.model_id == "m1"
    
    def test_allocate_budgets(self):
        controller = TrainingController({})
        profiles = [
            {"model_id": "m1", "priority": 1.0},
            {"model_id": "m2", "priority": 0.5},
        ]
        allocation = controller.allocate_budgets(profiles)
        assert len(allocation.model_budgets) == 2
    
    def test_evaluate_quality(self):
        controller = TrainingController({})
        controller.register_model("m1", {})
        profile = controller.evaluate_model_quality("m1", {"reasoning": 0.9}, {"reasoning": 0.92})
        assert profile.passes_all() is True
    
    def test_population_stats(self):
        controller = TrainingController({})
        controller.register_model("m1", {})
        stats = controller.get_population_stats()
        assert stats["total_models"] == 1


class TestKnowledgeReuse:
    def test_add_entry(self):
        store = KnowledgeStore({"reuse_policy": {"min_quality_threshold": 0.5, "require_verification": False}})
        entry = KnowledgeEntry(
            entry_id="e1",
            knowledge_type=KnowledgeType.VERIFIED_SOLUTION.value,
            source_model_id="m1",
            content={"text": "verified answer"},
            quality_score=0.9,
        )
        assert store.add_entry(entry) is True
    
    def test_reject_low_quality(self):
        store = KnowledgeStore({"reuse_policy": {"min_quality_threshold": 0.9, "require_verification": False}})
        entry = KnowledgeEntry(
            entry_id="e1",
            knowledge_type=KnowledgeType.VERIFIED_SOLUTION.value,
            source_model_id="m1",
            content={"text": "answer"},
            quality_score=0.5,
        )
        assert store.add_entry(entry) is False
    
    def test_get_entries_by_type(self):
        store = KnowledgeStore({"reuse_policy": {"min_quality_threshold": 0.0, "require_verification": False}})
        store.add_entry(KnowledgeEntry(entry_id="e1", knowledge_type=KnowledgeType.HARD_EXAMPLE.value, source_model_id="m1", content={}, quality_score=0.8))
        store.add_entry(KnowledgeEntry(entry_id="e2", knowledge_type=KnowledgeType.VERIFIED_SOLUTION.value, source_model_id="m1", content={}, quality_score=0.9))
        hard_examples = store.get_entries(knowledge_type=KnowledgeType.HARD_EXAMPLE.value)
        assert len(hard_examples) == 1
        assert hard_examples[0].entry_id == "e1"
    
    def test_reuse_manager(self):
        manager = KnowledgeReuseManager({"reuse_policy": {"enabled": True, "min_quality_threshold": 0.5, "require_verification": False}})
        entry = manager.record_knowledge("m1", KnowledgeType.VERIFIED_SOLUTION.value, {"text": "answer"}, 0.9)
        assert entry is not None
        assert len(manager.store.entries) == 1
    
    def test_contamination_check(self):
        store = KnowledgeStore({"reuse_policy": {"min_quality_threshold": 0.0, "contamination_check": True, "require_verification": False}})
        entry = KnowledgeEntry(
            entry_id="e1",
            knowledge_type=KnowledgeType.VERIFIED_SOLUTION.value,
            source_model_id="m1",
            content={"text": "answer"},
            quality_score=0.9,
            metadata={"is_benchmark": True},
        )
        assert store.add_entry(entry) is False


class TestLineage:
    def test_register_model(self):
        tracker = LineageTracker({})
        lineage = tracker.register_model("m1", generation=1, initialization_source="fresh")
        assert lineage.generation == 1
        assert lineage.initialization_source == "fresh"
        assert len(lineage.events) == 1
    
    def test_record_training_complete(self):
        tracker = LineageTracker({})
        tracker.register_model("m1", generation=1)
        tracker.record_training_complete("m1", "ckpt.pt", 0.9)
        lineage = tracker.get_lineage("m1")
        assert len(lineage.events) == 2
    
    def test_record_knowledge_reuse(self):
        tracker = LineageTracker({})
        tracker.register_model("m1", generation=1)
        tracker.register_model("m2", generation=1, parent_model_id="m1")
        tracker.record_knowledge_reuse("m2", "m1", "verified_solution")
        lineage = tracker.get_lineage("m2")
        assert "m1" in lineage.knowledge_sources
    
    def test_get_ancestors(self):
        tracker = LineageTracker({})
        tracker.register_model("m1", generation=0)
        tracker.register_model("m2", generation=1, parent_model_id="m1")
        tracker.register_model("m3", generation=2, parent_model_id="m2")
        ancestors = tracker.get_ancestors("m3")
        assert ancestors == ["m2", "m1"]
    
    def test_detect_common_ancestors(self):
        tracker = LineageTracker({})
        tracker.register_model("m1", generation=0)
        tracker.register_model("m2", generation=1, parent_model_id="m1")
        tracker.register_model("m3", generation=1, parent_model_id="m1")
        common = tracker.detect_common_ancestors(["m2", "m3"])
        assert "m1" in common["m2"]
        assert "m1" in common["m3"]
    
    def test_get_generation_lineage(self):
        tracker = LineageTracker({})
        tracker.register_model("m1", generation=1)
        tracker.register_model("m2", generation=2)
        gen1 = tracker.get_generation_lineage(1)
        assert len(gen1) == 1
        assert gen1[0].model_id == "m1"

    def test_check_convergence_detects_plateau(self):
        controller = TrainingController({})
        quality_history = {
            "model_00": [0.8, 0.81],
            "model_01": [0.4, 0.41],
        }
        assert controller._check_convergence(quality_history, window=2, threshold=0.05) is True

    def test_check_convergence_no_plateau(self):
        controller = TrainingController({})
        quality_history = {
            "model_00": [0.5, 0.7],
        }
        assert controller._check_convergence(quality_history, window=2, threshold=0.05) is False

    def test_decision_trace_contains_reason(self):
        controller = TrainingController({})
        budget = TrainingBudget(model_id="m1", max_wall_time_seconds=1000, max_tokens=500000, max_compute_flops=0.0005, quality_target=0.85)
        budget.current_quality = 0.5
        candidate = CandidateRecord(model_id="m1", overall_quality=0.5)
        trace = controller._build_decision_trace("m1", 0, budget, candidate, "RETAIN", "above floor")
        assert trace["decision"] == "RETAIN"
        assert trace["decision_reason"] == "above floor"
        assert "model_id" in trace
        assert "tokens_seen" in trace

    def test_hard_example_logs_flow_from_diagnostics(self):
        controller = TrainingController({})
        candidate = CandidateRecord(
            model_id="m1",
            overall_quality=0.2,
            failure_categories=["HIGH_VAL_LOSS", "VERIFICATION_WEAK"],
        )
        controller.model_contexts["m1"] = ModelTrainingContext(model_id="m1", candidate_record=candidate)
        trained_this_gen = ["m1"]
        hard_examples = []
        for mid in trained_this_gen:
            ctx = controller.model_contexts.get(mid)
            if ctx and ctx.candidate_record:
                for fc in ctx.candidate_record.failure_categories:
                    hard_examples.append({"model_id": mid, "failure_category": fc})
                    failure_log = FailureLog(
                        failure_id=f"{mid}_{fc}_test",
                        model_id=mid,
                        task_id=mid,
                        category=fc,
                        description=f"Training failure: {fc}",
                        metadata={"composite_score": ctx.candidate_record.composite_score},
                    )
                    controller.hard_example_miner.failure_logs.append(failure_log)
        assert len(hard_examples) == 2
        assert len(controller.hard_example_miner.failure_logs) == 2
