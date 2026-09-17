"""Task management for Prothynesis stateful models."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class TaskStateEnum(Enum):
    PENDING = "pending"
    PLANNING = "planning"
    REASONING = "reasoning"
    CRITIQUING = "critiquing"
    VERIFYING = "verifying"
    REVISING = "revising"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubtaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    MERGED = "merged"


@dataclass
class Subtask:
    """A subtask in the task graph."""
    subtask_id: str
    parent_task_id: str
    description: str
    status: str = SubtaskStatus.PENDING.value
    assigned_model: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "Subtask":
        return cls(**d)


@dataclass
class TaskState:
    """Complete state for a task."""
    task_id: str
    session_id: str
    objective: str
    constraints: List[str] = field(default_factory=list)
    subtasks: List[Subtask] = field(default_factory=list)
    current_subtask_id: Optional[str] = None
    plan: List[str] = field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    candidate_solutions: List[Dict[str, Any]] = field(default_factory=list)
    unresolved_issues: List[str] = field(default_factory=list)
    confidence: float = 0.0
    verification_state: str = "pending"
    peer_interactions: List[Dict[str, Any]] = field(default_factory=list)
    completion_state: str = TaskStateEnum.PENDING.value
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        data = asdict(self)
        data["completion_state"] = self.completion_state
        return data
    
    @classmethod
    def from_dict(cls, d: dict) -> "TaskState":
        return cls(**d)


class TaskManager:
    """
    Task manager for tracking and coordinating task state.
    
    Each model has a task manager that tracks:
    - current objective
    - subtasks
    - hypotheses
    - evidence
    - candidate solutions
    - peer interactions
    """
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self._tasks: Dict[str, TaskState] = {}
        self._active_task_id: Optional[str] = None
    
    def create_task(self, objective: str, constraints: Optional[List[str]] = None) -> TaskState:
        """Create a new task."""
        task_id = str(uuid.uuid4())
        
        task = TaskState(
            task_id=task_id,
            session_id=self.session_id,
            objective=objective,
            constraints=constraints or [],
        )
        
        self._tasks[task_id] = task
        self._active_task_id = task_id
        
        return task
    
    def get_task(self, task_id: str) -> Optional[TaskState]:
        """Get task state."""
        return self._tasks.get(task_id)
    
    def get_active_task(self) -> Optional[TaskState]:
        """Get currently active task."""
        if self._active_task_id:
            return self._tasks.get(self._active_task_id)
        return None
    
    def add_subtask(self, task_id: str, description: str) -> Optional[Subtask]:
        """Add a subtask to a task."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        subtask_id = str(uuid.uuid4())
        subtask = Subtask(
            subtask_id=subtask_id,
            parent_task_id=task_id,
            description=description,
        )
        
        task.subtasks.append(subtask)
        task.updated_at = datetime.now().isoformat()
        
        return subtask
    
    def assign_subtask(self, task_id: str, subtask_id: str, model_id: str) -> bool:
        """Assign a subtask to a model."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        for subtask in task.subtasks:
            if subtask.subtask_id == subtask_id:
                subtask.status = SubtaskStatus.ASSIGNED.value
                subtask.assigned_model = model_id
                task.updated_at = datetime.now().isoformat()
                return True
        
        return False
    
    def complete_subtask(self, task_id: str, subtask_id: str, result: Dict[str, Any]) -> bool:
        """Mark a subtask as completed."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        for subtask in task.subtasks:
            if subtask.subtask_id == subtask_id:
                subtask.status = SubtaskStatus.COMPLETED.value
                subtask.result = result
                subtask.confidence = result.get("confidence", 0.0)
                subtask.completed_at = datetime.now().isoformat()
                task.updated_at = datetime.now().isoformat()
                return True
        
        return False
    
    def add_hypothesis(self, task_id: str, hypothesis: Dict[str, Any]) -> bool:
        """Add a hypothesis to the task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.hypotheses.append(hypothesis)
        task.updated_at = datetime.now().isoformat()
        return True
    
    def add_evidence(self, task_id: str, evidence: Dict[str, Any]) -> bool:
        """Add evidence to the task."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.evidence.append(evidence)
        task.updated_at = datetime.now().isoformat()
        return True
    
    def add_candidate_solution(self, task_id: str, candidate: Dict[str, Any]) -> bool:
        """Add a candidate solution."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.candidate_solutions.append(candidate)
        task.updated_at = datetime.now().isoformat()
        return True
    
    def add_peer_interaction(self, task_id: str, interaction: Dict[str, Any]) -> bool:
        """Record a peer interaction."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.peer_interactions.append(interaction)
        task.updated_at = datetime.now().isoformat()
        return True
    
    def set_task_state(self, task_id: str, state: str) -> bool:
        """Update task state."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.completion_state = state
        task.updated_at = datetime.now().isoformat()
        return True
    
    def update_confidence(self, task_id: str, confidence: float) -> bool:
        """Update task confidence."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        
        task.confidence = confidence
        task.updated_at = datetime.now().isoformat()
        return True
    
    def get_task_summary(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task summary."""
        task = self._tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "objective": task.objective,
            "state": task.completion_state,
            "confidence": task.confidence,
            "num_subtasks": len(task.subtasks),
            "num_hypotheses": len(task.hypotheses),
            "num_evidence": len(task.evidence),
            "num_candidates": len(task.candidate_solutions),
            "num_peer_interactions": len(task.peer_interactions),
            "updated_at": task.updated_at,
        }
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize all task state."""
        return {
            "session_id": self.session_id,
            "active_task_id": self._active_task_id,
            "tasks": {tid: task.to_dict() for tid, task in self._tasks.items()},
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "TaskManager":
        """Deserialize task state."""
        manager = cls(session_id=data["session_id"])
        manager._active_task_id = data.get("active_task_id")
        
        for tid, task_data in data.get("tasks", {}).items():
            task = TaskState.from_dict(task_data)
            manager._tasks[tid] = task
        
        return manager
