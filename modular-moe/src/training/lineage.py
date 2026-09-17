"""Model lineage tracking for Prothynesis population training.

Tracks:
- parent_model_id
- parent_generation
- training lineage
- knowledge sources
- weight initialization source
- dataset profile
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class LineageEventType(Enum):
    """Types of lineage events."""
    INITIALIZATION = "initialization"
    TRAINING_START = "training_start"
    TRAINING_COMPLETE = "training_complete"
    CHECKPOINT_SAVED = "checkpoint_saved"
    KNOWLEDGE_REUSE = "knowledge_reuse"
    DISTILLATION = "distillation"
    HARD_EXAMPLE_LEARNED = "hard_example_learned"
    CURRICULUM_CHANGE = "curriculum_change"
    RESTART = "restart"
    EVOLUTION = "evolution"
    ARCHIVED = "archived"


@dataclass
class LineageEvent:
    """A single event in model lineage."""
    event_id: str
    event_type: str
    timestamp: float = field(default_factory=time.time)
    parent_model_id: Optional[str] = None
    parent_generation: Optional[int] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "parent_model_id": self.parent_model_id,
            "parent_generation": self.parent_generation,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class ModelLineage:
    """Complete lineage for a model."""
    model_id: str
    generation: int
    parent_model_id: Optional[str] = None
    parent_generation: Optional[int] = None
    initialization_source: str = "fresh"
    dataset_profile: Dict[str, Any] = field(default_factory=dict)
    knowledge_sources: List[str] = field(default_factory=list)
    weight_source: Optional[str] = None
    events: List[LineageEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    
    def add_event(self, event_type: str, description: str, metadata: Optional[Dict[str, Any]] = None) -> LineageEvent:
        """Add a lineage event."""
        event = LineageEvent(
            event_id=f"{self.model_id}_{event_type}_{int(time.time() * 1000)}",
            event_type=event_type,
            parent_model_id=self.parent_model_id,
            parent_generation=self.parent_generation,
            description=description,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "generation": self.generation,
            "parent_model_id": self.parent_model_id,
            "parent_generation": self.parent_generation,
            "initialization_source": self.initialization_source,
            "dataset_profile": self.dataset_profile,
            "knowledge_sources": self.knowledge_sources,
            "weight_source": self.weight_source,
            "events": [e.to_dict() for e in self.events],
            "created_at": self.created_at,
        }


class LineageTracker:
    """
    Tracks model lineage across generations.
    
    Supports:
    - Parent-child relationships
    - Knowledge source tracking
    - Weight source tracking
    - Event logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lineages: Dict[str, ModelLineage] = {}
    
    def register_model(self, model_id: str, generation: int, parent_model_id: Optional[str] = None, parent_generation: Optional[int] = None, initialization_source: str = "fresh", dataset_profile: Optional[Dict[str, Any]] = None) -> ModelLineage:
        """Register a new model lineage."""
        lineage = ModelLineage(
            model_id=model_id,
            generation=generation,
            parent_model_id=parent_model_id,
            parent_generation=parent_generation,
            initialization_source=initialization_source,
            dataset_profile=dataset_profile or {},
        )
        lineage.add_event(LineageEventType.INITIALIZATION.value, f"Model initialized ({initialization_source})")
        self.lineages[model_id] = lineage
        return lineage
    
    def get_lineage(self, model_id: str) -> Optional[ModelLineage]:
        """Get lineage for a model."""
        return self.lineages.get(model_id)
    
    def record_training_complete(self, model_id: str, checkpoint_path: str, quality_score: float) -> None:
        """Record training completion."""
        lineage = self.lineages.get(model_id)
        if lineage:
            lineage.add_event(
                LineageEventType.TRAINING_COMPLETE.value,
                f"Training complete. Quality: {quality_score:.4f}",
                {"checkpoint_path": checkpoint_path, "quality_score": quality_score},
            )
    
    def record_knowledge_reuse(self, model_id: str, source_model_id: str, knowledge_type: str) -> None:
        """Record knowledge reuse event."""
        lineage = self.lineages.get(model_id)
        if lineage:
            lineage.knowledge_sources.append(source_model_id)
            lineage.add_event(
                LineageEventType.KNOWLEDGE_REUSE.value,
                f"Reused {knowledge_type} from {source_model_id}",
                {"source_model_id": source_model_id, "knowledge_type": knowledge_type},
            )
    
    def record_weight_reuse(self, model_id: str, source_model_id: str) -> None:
        """Record weight reuse event."""
        lineage = self.lineages.get(model_id)
        if lineage:
            lineage.weight_source = source_model_id
            lineage.add_event(
                LineageEventType.KNOWLEDGE_REUSE.value,
                f"Reused weights from {source_model_id}",
                {"source_model_id": source_model_id},
            )
    
    def record_restart(self, model_id: str, run_id: int, reason: str) -> None:
        """Record restart event."""
        lineage = self.lineages.get(model_id)
        if lineage:
            lineage.add_event(
                LineageEventType.RESTART.value,
                f"Restarted run {run_id}: {reason}",
                {"run_id": run_id, "reason": reason},
            )
    
    def get_generation_lineage(self, generation: int) -> List[ModelLineage]:
        """Get all models in a generation."""
        return [lin for lin in self.lineages.values() if lin.generation == generation]
    
    def get_ancestors(self, model_id: str, max_depth: int = 10) -> List[str]:
        """Get ancestor model IDs."""
        ancestors = []
        current_id = model_id
        depth = 0
        
        while depth < max_depth:
            lineage = self.lineages.get(current_id)
            if lineage is None or lineage.parent_model_id is None:
                break
            ancestors.append(lineage.parent_model_id)
            current_id = lineage.parent_model_id
            depth += 1
        
        return ancestors
    
    def detect_common_ancestors(self, model_ids: List[str]) -> Dict[str, List[str]]:
        """Detect common ancestors among models."""
        ancestor_sets = {}
        for model_id in model_ids:
            ancestor_sets[model_id] = set(self.get_ancestors(model_id))
        
        # Find common ancestors
        if not ancestor_sets:
            return {}
        
        common = set.intersection(*ancestor_sets.values()) if len(ancestor_sets) > 1 else set()
        return {mid: list(common) for mid in model_ids}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get lineage statistics."""
        generation_counts: Dict[int, int] = {}
        source_counts: Dict[str, int] = {}
        for lineage in self.lineages.values():
            generation_counts[lineage.generation] = generation_counts.get(lineage.generation, 0) + 1
            source_counts[lineage.initialization_source] = source_counts.get(lineage.initialization_source, 0) + 1
        
        return {
            "total_models": len(self.lineages),
            "generation_distribution": generation_counts,
            "initialization_sources": source_counts,
        }
