"""Tests for Collective Reasoning Workspace (CRW)."""

from __future__ import annotations

import threading
import time

import pytest

from src.momm.collective_workspace import (
    CollectiveEvent,
    CollectiveEventType,
    CollectiveReasoningWorkspace,
    CollectiveWorkingState,
    ContextSlot,
    LongTermMemoryContext,
    MemoryReference,
    WorkspaceSnapshot,
    WorkingHypothesis,
)


def _make_workspace(**kwargs):
    defaults = dict(
        task_id="task_001",
        session_id="session_001",
        default_token_budget=2048,
        max_events=100,
        max_slots=4,
        max_l2_refs=100,
        persistence_enabled=False,
    )
    defaults.update(kwargs)
    return CollectiveReasoningWorkspace(**defaults)


def test_workspace_creation():
    ws = _make_workspace()
    assert ws.task_id == "task_001"
    assert ws.session_id == "session_001"
    assert ws.default_token_budget == 2048


def test_emit_proposal_creates_hypothesis():
    ws = _make_workspace()
    event = ws.emit(
        CollectiveEventType.PROPOSAL,
        sender_id="solver_a",
        content="Hypothesis X",
        confidence=0.8,
    )
    assert event.event_id is not None
    assert event.event_type == "proposal"
    assert len(ws.get_events()) == 1
    assert len(ws.get_working_state().active_hypotheses) == 1


def test_emit_critique_records_disagreement():
    ws = _make_workspace()
    proposal = ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    critique = ws.emit(
        CollectiveEventType.CRITIQUE,
        sender_id="solver_b",
        content="H1 is wrong",
        target_ids=["solver_a"],
        confidence=0.7,
        metadata={"hypothesis_id": list(ws.get_working_state().active_hypotheses.keys())[0]},
    )
    assert critique.event_type == "critique"
    disagreements = ws.get_disagreements()
    assert len(disagreements) == 1


def test_emit_evidence_attaches_to_hypothesis():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    hyp_id = list(ws.get_working_state().active_hypotheses.keys())[0]
    ws.emit(
        CollectiveEventType.EVIDENCE,
        sender_id="solver_c",
        content="Evidence E1",
        evidence_refs=["ev_001"],
        metadata={"hypothesis_id": hyp_id},
    )
    hyp = ws.get_hypothesis(hyp_id)
    assert hyp is not None
    assert "ev_001" in hyp.evidence


def test_emit_revision_updates_hypothesis():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    hyp_id = list(ws.get_working_state().active_hypotheses.keys())[0]
    ws.emit(
        CollectiveEventType.REVISION,
        sender_id="solver_a",
        content="H1 revised",
        confidence=0.9,
        metadata={"hypothesis_id": hyp_id},
    )
    hyp = ws.get_hypothesis(hyp_id)
    assert hyp.content == "H1 revised"
    assert hyp.revision_count == 1


def test_emit_verification_adjusts_confidence():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.5)
    hyp_id = list(ws.get_working_state().active_hypotheses.keys())[0]
    ws.emit(
        CollectiveEventType.VERIFICATION,
        sender_id="solver_b",
        content="verified",
        confidence=0.9,
        metadata={"hypothesis_id": hyp_id, "verified": True},
    )
    hyp = ws.get_hypothesis(hyp_id)
    assert hyp.confidence >= 0.5


def test_emit_agreement_advances_consensus():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.6)
    hyp_id = list(ws.get_working_state().active_hypotheses.keys())[0]
    ws.emit(
        CollectiveEventType.AGREEMENT,
        sender_id="solver_b",
        content="agree",
        metadata={"hypothesis_id": hyp_id},
    )
    consensus = ws.get_consensus()
    assert consensus is not None
    assert consensus["hypothesis_id"] == hyp_id


def test_emit_disagreement_records_warning_and_disagreement():
    ws = _make_workspace()
    ws.emit(
        CollectiveEventType.DISAGREEMENT,
        sender_id="solver_a",
        content="disagree",
        target_ids=["solver_b"],
    )
    assert len(ws.get_disagreements()) == 1


def test_emit_question_adds_open_question():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.QUESTION, sender_id="solver_a", content="Why X?")
    assert "Why X?" in ws.get_open_questions()


def test_emit_recruitment_and_stop_requests():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.RECRUITMENT_REQUEST, sender_id="solver_a", content="need help", confidence=0.9)
    ws.emit(CollectiveEventType.STOP_REQUEST, sender_id="solver_b", content="stop")
    assert len(ws.get_recruitment_requests()) == 1
    assert "solver_b" in ws.get_stop_requests()


def test_emit_summary_sets_summary():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.SUMMARY, sender_id="solver_a", content="All done")
    assert ws.get_summary() == "All done"


def test_event_filtering():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    ws.emit(CollectiveEventType.CRITIQUE, sender_id="solver_b", content="bad")
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_c", content="H2")
    proposals = ws.get_events(event_type=CollectiveEventType.PROPOSAL)
    assert len(proposals) == 2
    from_a = ws.get_events(sender_id="solver_a")
    assert len(from_a) == 1


def test_get_event_by_id():
    ws = _make_workspace()
    event = ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    found = ws.get_event_by_id(event.event_id)
    assert found is not None
    assert found.event_id == event.event_id


def test_l0_stream_summary():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    stream = ws.get_l0_stream()
    assert len(stream) == 1
    assert stream[0]["event_type"] == "proposal"


def test_l1_summary():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    summary = ws.get_working_state()
    assert len(summary.active_hypotheses) == 1
    assert summary.consensus_confidence == 0.0


def test_l2_memory_references():
    ws = _make_workspace()
    ref = ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    assert ref.ref_id is not None
    refs = ws.get_memory_references(min_relevance=0.8)
    assert len(refs) == 1
    assert refs[0].memory_id == "mem_001"


def test_l2_context():
    ws = _make_workspace()
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    ctx = ws.get_l2_context()
    assert ctx["total_references"] == 1


def test_acquire_slot():
    ws = _make_workspace(max_slots=2)
    slot = ws.acquire_slot("solver_a", token_budget=1024)
    assert slot.model_id == "solver_a"
    assert slot.token_budget == 1024
    assert slot.is_active()


def test_acquire_slot_duplicate_raises():
    ws = _make_workspace()
    ws.acquire_slot("solver_a")
    with pytest.raises(RuntimeError):
        ws.acquire_slot("solver_a")


def test_acquire_slot_capacity_raises():
    ws = _make_workspace(max_slots=1)
    ws.acquire_slot("solver_a")
    with pytest.raises(RuntimeError):
        ws.acquire_slot("solver_b")


def test_release_slot():
    ws = _make_workspace()
    slot = ws.acquire_slot("solver_a")
    released = ws.release_slot("solver_a")
    assert released is not None
    assert released.slot_id == slot.slot_id
    assert not released.is_active()


def test_release_slot_missing_returns_none():
    ws = _make_workspace()
    assert ws.release_slot("solver_a") is None


def test_handoff_slot():
    ws = _make_workspace()
    ws.acquire_slot("solver_a")
    handed = ws.handoff_slot("solver_a", "solver_b")
    assert handed is not None
    assert handed.handoff_to == "solver_b"
    assert ws.get_slot("solver_b") is not None
    assert ws.get_slot("solver_a") is None


def test_handoff_slot_target_occupied_raises():
    ws = _make_workspace()
    ws.acquire_slot("solver_a")
    ws.acquire_slot("solver_b")
    with pytest.raises(RuntimeError):
        ws.handoff_slot("solver_a", "solver_b")


def test_get_slot():
    ws = _make_workspace()
    assert ws.get_slot("solver_a") is None
    slot = ws.acquire_slot("solver_a")
    assert ws.get_slot("solver_a") is not None


def test_get_active_slots():
    ws = _make_workspace()
    ws.acquire_slot("solver_a")
    ws.acquire_slot("solver_b")
    assert len(ws.get_active_slots()) == 2


def test_record_slot_usage():
    ws = _make_workspace()
    ws.acquire_slot("solver_a")
    ws.record_slot_usage("solver_a", 100)
    slot = ws.get_slot("solver_a")
    assert slot.tokens_used == 100


def test_build_context():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    ctx = ws.build_context(for_model_id="solver_a", max_tokens=1024)
    assert ctx["task_id"] == "task_001"
    assert "context" in ctx
    assert ctx["for_model_id"] == "solver_a"


def test_build_context_excludes_l0_when_disabled():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    ctx = ws.build_context(for_model_id="solver_a", include_l0=False)
    assert ctx["context"] == ""


def test_build_context_excludes_l2_when_disabled():
    ws = _make_workspace()
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    ctx = ws.build_context(for_model_id="solver_a", include_l2=False)
    assert ctx["context"] == ""


def test_snapshot_and_restore():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    snap = ws.snapshot()
    assert snap.snapshot_id is not None
    assert len(ws.list_snapshots()) == 1

    ws.emit(CollectiveEventType.CRITIQUE, sender_id="solver_b", content="bad")
    ws.restore(snap.snapshot_id)
    assert len(ws.get_events()) == 1
    assert ws.get_events()[0].event_type == "proposal"


def test_restore_missing_snapshot_raises():
    ws = _make_workspace()
    with pytest.raises(KeyError):
        ws.restore("missing_snap")


def test_get_snapshot():
    ws = _make_workspace()
    snap = ws.snapshot()
    fetched = ws.get_snapshot(snap.snapshot_id)
    assert fetched is not None
    assert fetched.snapshot_id == snap.snapshot_id


def test_list_snapshots():
    ws = _make_workspace()
    ws.snapshot()
    ws.snapshot()
    assert len(ws.list_snapshots()) == 2


def test_serialize_deserialize():
    ws = _make_workspace(persistence_enabled=False)
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    data = ws.serialize()
    restored = CollectiveReasoningWorkspace.deserialize(data)
    assert restored.task_id == ws.task_id
    assert restored.session_id == ws.session_id
    assert len(restored.get_events()) == 1
    assert len(restored.get_memory_references()) == 1


def test_clear():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    ws.clear()
    assert len(ws.get_events()) == 0
    assert len(ws.get_memory_references()) == 0
    assert len(ws.get_active_slots()) == 0


def test_get_stats():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    stats = ws.get_stats()
    assert stats["l0_event_count"] == 1
    assert stats["l1_hypothesis_count"] == 1


def test_layer0_summary():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1")
    summary = ws.get_layer0_summary(limit=10)
    assert len(summary) == 1
    assert summary[0]["event_type"] == "proposal"


def test_layer1_summary():
    ws = _make_workspace()
    ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content="H1", confidence=0.8)
    summary = ws.get_layer1_summary()
    assert summary["active_hypotheses"] == 1
    assert summary["consensus_confidence"] == 0.0


def test_layer2_summary():
    ws = _make_workspace()
    ws.integrate_memory(memory_id="mem_001", memory_type="task", relevance_score=0.9)
    summary = ws.get_layer2_summary(limit=10)
    assert len(summary) == 1


def test_thread_safety_emit():
    ws = _make_workspace()
    errors = []

    def _emit(sender):
        for i in range(10):
            try:
                ws.emit(CollectiveEventType.PROPOSAL, sender_id=sender, content=f"H_{i}")
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=_emit, args=(f"solver_{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(ws.get_events()) == 40


def test_thread_safety_slots():
    ws = _make_workspace(max_slots=10)
    errors = []

    def _acquire_release(model_id):
        for _ in range(5):
            try:
                slot = ws.acquire_slot(model_id)
                time.sleep(0.001)
                ws.release_slot(model_id)
                assert slot is not None
            except Exception as exc:
                errors.append(exc)

    threads = [threading.Thread(target=_acquire_release, args=(f"solver_{i}",)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors


def test_prune_events():
    ws = _make_workspace(max_events=5)
    for i in range(10):
        ws.emit(CollectiveEventType.PROPOSAL, sender_id="solver_a", content=f"H{i}")
    assert len(ws.get_events()) == 5


def test_prune_l2():
    ws = _make_workspace(max_l2_refs=3)
    for i in range(5):
        ws.integrate_memory(memory_id=f"mem_{i}", memory_type="task", relevance_score=float(i))
    assert len(ws.get_memory_references()) <= 3


def test_request_recruitment():
    ws = _make_workspace()
    result = ws.request_recruitment(sender_id="solver_a", reason="hard task", urgency=0.9)
    assert "event_id" in result
    assert len(ws.get_recruitment_requests()) == 1


def test_record_failure():
    ws = _make_workspace()
    ws.record_failure(model_id="solver_a", failure_category="hallucination", description="wrong answer", confidence=0.8)
    assert len(ws.get_warnings()) == 1


def test_reflection_without_engine():
    ws = _make_workspace()
    result = ws.reflect(model_id="solver_a", confidence=0.8, reasoning_quality=0.7)
    assert result is None


def test_verify_candidate_without_verifier():
    ws = _make_workspace()
    result = ws.verify_candidate(candidate_id="c1", content="answer", evidence=[], verifier_id="solver_b")
    assert result is None


def test_meta_verify_without_meta_verifier():
    ws = _make_workspace()
    result = ws.meta_verify(candidate_id="c1", content="answer", evidence=[], prior_verifications=[], verifier_id="solver_b")
    assert result is None


def test_dataclass_roundtrip():
    event = CollectiveEvent(
        event_id="e1",
        event_type="proposal",
        task_id="t1",
        session_id="s1",
        sender_id="m1",
        target_ids=[],
        content="hello",
        confidence=0.9,
    )
    d = event.to_dict()
    restored = CollectiveEvent.from_dict(d)
    assert restored.event_id == "e1"
    assert restored.event_type == "proposal"

    hyp = WorkingHypothesis(hypothesis_id="h1", content="H1", confidence=0.8)
    d = hyp.to_dict()
    restored_hyp = WorkingHypothesis.from_dict(d)
    assert restored_hyp.hypothesis_id == "h1"

    slot = ContextSlot(slot_id="slot1", model_id="m1", task_id="t1", session_id="s1", acquired_at="now")
    d = slot.to_dict()
    restored_slot = ContextSlot.from_dict(d)
    assert restored_slot.slot_id == "slot1"

    snap = WorkspaceSnapshot(snapshot_id="snap1", task_id="t1", session_id="s1")
    d = snap.to_dict()
    restored_snap = WorkspaceSnapshot.from_dict(d)
    assert restored_snap.snapshot_id == "snap1"

    state = CollectiveWorkingState(task_id="t1", session_id="s1")
    d = state.to_dict()
    restored_state = CollectiveWorkingState.from_dict(d)
    assert restored_state.task_id == "t1"

    l2 = LongTermMemoryContext(task_id="t1", session_id="s1")
    d = l2.to_dict()
    restored_l2 = LongTermMemoryContext.from_dict(d)
    assert restored_l2.task_id == "t1"

    ref = MemoryReference(ref_id="r1", memory_id="m1", memory_type="task", task_id="t1", session_id="s1")
    d = ref.to_dict()
    restored_ref = MemoryReference.from_dict(d)
    assert restored_ref.ref_id == "r1"


def test_integrate_with_memory_without_manager():
    ws = _make_workspace()
    record = ws.integrate_with_memory(content="test", memory_type="working")
    assert record is None


def test_close_releases_slots():
    ws = _make_workspace()
    ws.acquire_slot("solver_a")
    ws.acquire_slot("solver_b")
    ws.close()
    assert len(ws.get_active_slots()) == 0
