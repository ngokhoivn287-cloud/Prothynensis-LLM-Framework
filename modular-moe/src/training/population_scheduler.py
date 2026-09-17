"""Population training scheduler for Prothynesis.

Manages scheduling of thousands of model training jobs with:
- Priority queues
- Resource awareness
- Deadline tracking
- Status management
- Resumability
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from queue import PriorityQueue


class ModelTrainingStatus(Enum):
    """Model training status."""
    QUEUED = "queued"
    TRAINING = "training"
    EVALUATING = "evaluating"
    RESTARTING = "restarting"
    PAUSED = "paused"
    COMPLETE = "complete"
    REJECTED = "rejected"
    ARCHIVED = "archived"


@dataclass(order=True)
class ModelTrainingRecord:
    """Record for a model in the training queue."""
    priority: float
    model_id: str
    status: str = field(default=ModelTrainingStatus.QUEUED.value, compare=False)
    budget: Optional[Dict[str, Any]] = field(default=None, compare=False)
    deadline_timestamp: Optional[float] = field(default=None, compare=False)
    hardware_requirement: Dict[str, Any] = field(default_factory=dict, compare=False)
    quality: float = 0.0
    marginal_gain: float = 0.0
    current_stage: str = "warmup"
    attempts: int = 0
    max_attempts: int = 3
    queued_at: float = field(default_factory=time.time, compare=False)
    started_at: Optional[float] = field(default=None, compare=False)
    completed_at: Optional[float] = field(default=None, compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "priority": self.priority,
            "status": self.status,
            "budget": self.budget,
            "deadline_timestamp": self.deadline_timestamp,
            "hardware_requirement": self.hardware_requirement,
            "quality": self.quality,
            "marginal_gain": self.marginal_gain,
            "current_stage": self.current_stage,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }


class PopulationScheduler:
    """
    Scheduler for population training.
    
    Manages a priority queue of models to train, with support for:
    - Deadline-aware scheduling
    - Resource-aware scheduling
    - Resumability
    - Status tracking
    - Retry logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._queue: PriorityQueue = PriorityQueue()
        self._records: Dict[str, ModelTrainingRecord] = {}
        self._lock = threading.Lock()
        self._active_trainers: Dict[str, Any] = {}
    
    def enqueue(self, record: ModelTrainingRecord) -> None:
        """Add a model to the training queue."""
        with self._lock:
            self._records[record.model_id] = record
            self._queue.put(record)
    
    def enqueue_batch(self, records: List[ModelTrainingRecord]) -> None:
        """Add multiple models to the queue."""
        for record in records:
            self.enqueue(record)
    
    def dequeue(self) -> Optional[ModelTrainingRecord]:
        """Get the highest-priority model ready for training."""
        with self._lock:
            while not self._queue.empty():
                record = self._queue.get()
                if record.status == ModelTrainingStatus.QUEUED.value:
                    # Check deadline
                    if record.deadline_timestamp is not None and time.time() > record.deadline_timestamp:
                        record.status = ModelTrainingStatus.REJECTED.value
                        continue
                    record.status = ModelTrainingStatus.TRAINING.value
                    record.started_at = time.time()
                    return record
            return None
    
    def mark_complete(self, model_id: str, quality: float = 0.0) -> None:
        """Mark a model as complete."""
        with self._lock:
            if model_id in self._records:
                record = self._records[model_id]
                record.status = ModelTrainingStatus.COMPLETE.value
                record.completed_at = time.time()
                record.quality = quality
    
    def mark_failed(self, model_id: str, retry: bool = True) -> None:
        """Mark a model as failed and optionally requeue."""
        with self._lock:
            if model_id in self._records:
                record = self._records[model_id]
                record.attempts += 1
                if retry and record.attempts < record.max_attempts:
                    record.status = ModelTrainingStatus.QUEUED.value
                    record.started_at = None
                    self._queue.put(record)
                else:
                    record.status = ModelTrainingStatus.REJECTED.value
                    record.completed_at = time.time()
    
    def get_status(self, model_id: str) -> Optional[str]:
        """Get current status of a model."""
        with self._lock:
            record = self._records.get(model_id)
            return record.status if record else None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            status_counts: Dict[str, int] = {}
            for record in self._records.values():
                status_counts[record.status] = status_counts.get(record.status, 0) + 1
            
            return {
                "total": len(self._records),
                "queued": status_counts.get(ModelTrainingStatus.QUEUED.value, 0),
                "training": status_counts.get(ModelTrainingStatus.TRAINING.value, 0),
                "complete": status_counts.get(ModelTrainingStatus.COMPLETE.value, 0),
                "rejected": status_counts.get(ModelTrainingStatus.REJECTED.value, 0),
                "archived": status_counts.get(ModelTrainingStatus.ARCHIVED.value, 0),
            }
    
    def get_models_by_status(self, status: ModelTrainingStatus) -> List[ModelTrainingRecord]:
        """Get all models with a given status."""
        with self._lock:
            return [r for r in self._records.values() if r.status == status.value]
    
    def get_deadline_models(self, within_seconds: float) -> List[ModelTrainingRecord]:
        """Get models with deadlines within the given time window."""
        now = time.time()
        with self._lock:
            return [
                r for r in self._records.values()
                if r.deadline_timestamp is not None and 0 < r.deadline_timestamp - now <= within_seconds
            ]
    
    def save_state(self, path: str) -> None:
        """Save scheduler state for resumability."""
        import json
        with self._lock:
            data = {
                "records": {mid: record.to_dict() for mid, record in self._records.items()},
                "timestamp": time.time(),
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
    
    def load_state(self, path: str) -> None:
        """Load scheduler state."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        
        with self._lock:
            self._records.clear()
            self._queue = PriorityQueue()
            
            for model_id, record_data in data.get("records", {}).items():
                record = ModelTrainingRecord(
                    priority=record_data["priority"],
                    model_id=record_data["model_id"],
                    status=record_data.get("status", ModelTrainingStatus.QUEUED.value),
                    deadline_timestamp=record_data.get("deadline_timestamp"),
                    hardware_requirement=record_data.get("hardware_requirement", {}),
                    quality=record_data.get("quality", 0.0),
                    marginal_gain=record_data.get("marginal_gain", 0.0),
                    current_stage=record_data.get("current_stage", "warmup"),
                    attempts=record_data.get("attempts", 0),
                    max_attempts=record_data.get("max_attempts", 3),
                    queued_at=record_data.get("queued_at", time.time()),
                    started_at=record_data.get("started_at"),
                    completed_at=record_data.get("completed_at"),
                    metadata=record_data.get("metadata", {}),
                )
                self._records[model_id] = record
                if record.status == ModelTrainingStatus.QUEUED.value:
                    self._queue.put(record)
