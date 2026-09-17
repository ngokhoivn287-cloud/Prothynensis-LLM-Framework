"""Tests for Prothynesis memory, task, communication, and collaboration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_memory_manager_working_memory():
    """Test working memory creation and retrieval."""
    from momm.memory import MemoryManager, MemoryType
    
    manager = MemoryManager(
        model_id="solver_0001",
        task_id="task_001",
        session_id="session_001",
        max_working_memory=10,
    )
    
    # Create working memories
    for i in range(5):
        manager.create_memory(
            memory_type=MemoryType.WORKING.value,
            content=f"working fact {i}",
            importance=0.5 + i * 0.1,
        )
    
    stats = manager.get_stats()
    assert stats["by_type"][MemoryType.WORKING.value] == 5
    
    # Retrieve
    memories = manager.retrieve(memory_type=MemoryType.WORKING.value)
    assert len(memories) == 5
    assert memories[0].importance >= memories[-1].importance
    
    print("  PASS: Working memory works")


def test_memory_manager_task_memory():
    """Test task memory."""
    from momm.memory import MemoryManager, MemoryType
    
    manager = MemoryManager(
        model_id="solver_0001",
        task_id="task_001",
        session_id="session_001",
    )
    
    # Create task memories
    manager.create_memory(
        memory_type=MemoryType.TASK.value,
        content="objective: solve math problem",
        importance=0.9,
        tags=["objective"],
    )
    manager.create_memory(
        memory_type=MemoryType.TASK.value,
        content="subtask: verify approach",
        importance=0.7,
        tags=["subtask"],
    )
    
    memories = manager.retrieve(memory_type=MemoryType.TASK.value, tags=["objective"])
    assert len(memories) == 1
    assert "objective" in memories[0].content
    
    print("  PASS: Task memory works")


def test_memory_manager_peer_memory():
    """Test peer memory."""
    from momm.memory import MemoryManager, MemoryType
    
    manager = MemoryManager(
        model_id="solver_0001",
        task_id="task_001",
        session_id="session_001",
    )
    
    # Receive peer information
    manager.create_memory(
        memory_type=MemoryType.PEER.value,
        content="solver_0002: evidence E1",
        source="solver_0002",
        importance=0.8,
        memory_refs=["evidence_E1"],
    )
    
    memories = manager.retrieve(memory_type=MemoryType.PEER.value)
    assert len(memories) == 1
    assert memories[0].source == "solver_0002"
    
    print("  PASS: Peer memory works")


def test_memory_manager_serialization():
    """Test memory serialization."""
    from momm.memory import MemoryManager, MemoryType
    
    manager = MemoryManager(
        model_id="solver_0001",
        task_id="task_001",
        session_id="session_001",
    )
    
    manager.create_memory(
        memory_type=MemoryType.WORKING.value,
        content="test fact",
        importance=0.8,
    )
    
    # Serialize
    data = manager.serialize()
    assert "memories" in data
    assert len(data["memories"]) == 1
    
    # Deserialize
    restored = MemoryManager.deserialize(data)
    assert restored.model_id == manager.model_id
    assert restored.task_id == manager.task_id
    assert len(restored._memories) == 1
    
    print("  PASS: Memory serialization works")


def test_memory_manager_update_and_delete():
    """Test memory update and delete."""
    from momm.memory import MemoryManager, MemoryType
    
    manager = MemoryManager(
        model_id="solver_0001",
        task_id="task_001",
        session_id="session_001",
    )
    
    memory = manager.create_memory(
        memory_type=MemoryType.WORKING.value,
        content="original",
        importance=0.5,
    )
    
    # Update
    updated = manager.update_memory(memory.memory_id, content="updated", importance=0.9)
    assert updated is not None
    assert updated.content == "updated"
    assert updated.version == 2
    
    # Delete
    assert manager.delete_memory(memory.memory_id)
    assert manager.retrieve(memory_type=MemoryType.WORKING.value) == []
    
    print("  PASS: Memory update and delete work")


def test_task_manager_lifecycle():
    """Test task manager lifecycle."""
    from momm.task import TaskManager, TaskStateEnum
    
    manager = TaskManager(session_id="session_001")
    
    # Create task
    task = manager.create_task(objective="Solve problem X", constraints=["time limit"])
    assert task.task_id is not None
    assert task.objective == "Solve problem X"
    assert task.completion_state == TaskStateEnum.PENDING.value
    
    # Add subtask
    subtask = manager.add_subtask(task.task_id, "Analyze problem")
    assert subtask is not None
    assert subtask.description == "Analyze problem"
    
    # Assign subtask
    assert manager.assign_subtask(task.task_id, subtask.subtask_id, "solver_0001")
    
    # Complete subtask
    assert manager.complete_subtask(task.task_id, subtask.subtask_id, {"result": "done"})
    
    # Add hypothesis
    assert manager.add_hypothesis(task.task_id, {"hypothesis": "H1", "confidence": 0.8})
    
    # Add evidence
    assert manager.add_evidence(task.task_id, {"evidence": "E1", "source": "solver_0002"})
    
    # Update state
    assert manager.set_task_state(task.task_id, TaskStateEnum.REASONING.value)
    assert manager.update_confidence(task.task_id, 0.85)
    
    # Get summary
    summary = manager.get_task_summary(task.task_id)
    assert summary is not None
    assert summary["state"] == TaskStateEnum.REASONING.value
    assert summary["confidence"] == 0.85
    
    print("  PASS: Task manager lifecycle works")


def test_task_manager_serialization():
    """Test task manager serialization."""
    from momm.task import TaskManager, TaskState
    
    manager = TaskManager(session_id="session_001")
    task = manager.create_task(objective="Test task")
    manager.add_subtask(task.task_id, "subtask 1")
    manager.add_hypothesis(task.task_id, {"h": 1})
    
    # Serialize
    data = manager.serialize()
    assert len(data["tasks"]) == 1
    
    # Deserialize
    restored = TaskManager.deserialize(data)
    assert restored.session_id == manager.session_id
    assert len(restored._tasks) == 1
    assert list(restored._tasks.values())[0].objective == "Test task"
    
    print("  PASS: Task manager serialization works")


def test_peer_communication_protocol():
    """Test peer communication protocol."""
    from momm.communication import PeerCommunicationProtocol, MessageType
    
    protocol = PeerCommunicationProtocol(
        model_id="solver_0001",
        task_id="task_001",
    )
    
    # Send critique
    critique = protocol.send_critique(
        target="solver_0002",
        claim="approach is flawed",
        evidence="evidence_E1",
        confidence=0.8,
    )
    assert critique.message_type == MessageType.CRITIQUE.value
    assert critique.target == "solver_0002"
    
    # Send evidence
    evidence = protocol.send_evidence(
        target="solver_0002",
        evidence_summary="supporting data",
        confidence=0.9,
    )
    assert evidence.message_type == MessageType.EVIDENCE.value
    
    # Send verification request
    verify_req = protocol.send_verification_request(
        target="solver_0003",
        claim="result X",
        evidence="evidence_E2",
    )
    assert verify_req.message_type == MessageType.VERIFICATION_REQUEST.value
    
    # Get messages
    critiques = protocol.get_critiques()
    assert len(critiques) == 1
    
    verifications = protocol.get_verifications()
    assert len(verifications) == 0  # No verification results yet
    
    # Stats
    stats = protocol.get_message_stats()
    assert stats["total_messages"] == 3
    assert stats["unique_targets"] == 2
    
    print("  PASS: Peer communication protocol works")


def test_collaborative_synthesis():
    """Test collaborative synthesis."""
    from momm.collaboration import CollaborativeSynthesizer, SynthesisStrategy
    
    synthesizer = CollaborativeSynthesizer(
        task_id="task_001",
        strategy=SynthesisStrategy.EVIDENCE_WEIGHTED.value,
    )
    
    # Add candidates from different models
    c1 = synthesizer.add_candidate(
        model_id="solver_0001",
        content="candidate A",
        confidence=0.7,
        evidence=["E1", "E2"],
    )
    c2 = synthesizer.add_candidate(
        model_id="solver_0002",
        content="candidate B",
        confidence=0.9,
        evidence=["E3", "E4", "E5"],
    )
    c3 = synthesizer.add_candidate(
        model_id="solver_0003",
        content="candidate C",
        confidence=0.6,
        evidence=["E6"],
    )
    
    # Add critiques
    synthesizer.add_critique(c1.candidate_id, "flawed", "solver_0003")
    synthesizer.add_critique(c3.candidate_id, "weak", "solver_0001")
    
    # Add verifications
    synthesizer.add_verification(c2.candidate_id, "claim_B", True)
    synthesizer.add_verification(c2.candidate_id, "claim_B2", True)
    
    # Synthesize
    result = synthesizer.synthesize()
    
    assert result.task_id == "task_001"
    assert result.total_candidates == 3
    assert len(result.participating_models) == 3
    # Candidate B should win (higher confidence, more evidence, verifications)
    assert result.final_candidate == "candidate B"
    assert result.confidence > 0.0
    
    print(f"  PASS: Synthesis selected '{result.final_candidate}' "
          f"(confidence={result.confidence:.2f}, consensus={result.consensus_score:.2f})")


def test_collaborative_synthesis_revision():
    """Test synthesis with revisions."""
    from momm.collaboration import CollaborativeSynthesizer
    
    synthesizer = CollaborativeSynthesizer(task_id="task_002")
    
    c1 = synthesizer.add_candidate(
        model_id="solver_0001",
        content="original answer",
        confidence=0.5,
    )
    
    # Model revises after peer feedback
    synthesizer.add_revision(c1.candidate_id, "revised answer", "solver_0001")
    
    result = synthesizer.synthesize()
    assert result.final_candidate == "revised answer"
    
    print("  PASS: Collaborative synthesis with revision works")


def test_memory_isolation():
    """Test that task memory is isolated between tasks."""
    from momm.memory import MemoryManager, MemoryType
    
    # Task A memory
    manager_a = MemoryManager(
        model_id="solver_0001",
        task_id="task_A",
        session_id="session_001",
    )
    manager_a.create_memory(
        memory_type=MemoryType.TASK.value,
        content="Task A secret",
        importance=0.9,
        tags=["task_a"],
    )
    
    # Task B memory
    manager_b = MemoryManager(
        model_id="solver_0001",
        task_id="task_B",
        session_id="session_001",
    )
    manager_b.create_memory(
        memory_type=MemoryType.TASK.value,
        content="Task B secret",
        importance=0.9,
        tags=["task_b"],
    )
    
    # Verify isolation
    task_a_memories = manager_a.retrieve(memory_type=MemoryType.TASK.value)
    task_b_memories = manager_b.retrieve(memory_type=MemoryType.TASK.value)
    
    assert len(task_a_memories) == 1
    assert len(task_b_memories) == 1
    assert task_a_memories[0].content == "Task A secret"
    assert task_b_memories[0].content == "Task B secret"
    
    print("  PASS: Memory isolation between tasks verified")


def test_message_schema():
    """Test peer message schema."""
    from momm.communication import PeerMessage
    
    msg = PeerMessage(
        message_id="msg_001",
        message_type="CRITIQUE",
        task_id="task_001",
        sender="solver_0001",
        target="solver_0002",
        content_summary="approach is flawed",
        evidence_refs=["E1"],
        memory_refs=["M1"],
        confidence=0.82,
    )
    
    data = msg.to_dict()
    assert data["message_type"] == "CRITIQUE"
    assert data["sender"] == "solver_0001"
    assert data["target"] == "solver_0002"
    assert data["confidence"] == 0.82
    
    restored = PeerMessage.from_dict(data)
    assert restored.message_id == "msg_001"
    assert restored.content_summary == "approach is flawed"
    
    print("  PASS: Message schema works")


def run_all_tests():
    """Run all new tests."""
    tests = [
        ("Memory Working", test_memory_manager_working_memory),
        ("Memory Task", test_memory_manager_task_memory),
        ("Memory Peer", test_memory_manager_peer_memory),
        ("Memory Serialization", test_memory_manager_serialization),
        ("Memory Update/Delete", test_memory_manager_update_and_delete),
        ("Task Manager Lifecycle", test_task_manager_lifecycle),
        ("Task Manager Serialization", test_task_manager_serialization),
        ("Peer Communication", test_peer_communication_protocol),
        ("Collaborative Synthesis", test_collaborative_synthesis),
        ("Synthesis with Revision", test_collaborative_synthesis_revision),
        ("Memory Isolation", test_memory_isolation),
        ("Message Schema", test_message_schema),
    ]
    
    print("=" * 60)
    print("PROTHYNESIS MEMORY/COLLABORATION TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
