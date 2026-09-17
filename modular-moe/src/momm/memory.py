"""Memory subsystem for Prothynesis stateful models."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class MemoryType(Enum):
    WORKING = "working"
    TASK = "task"
    EPISODIC = "episodic"
    PEER = "peer"
    LONG_TERM = "long_term"


@dataclass
class MemoryRecord:
    """A single memory record."""
    memory_id: str
    task_id: str
    session_id: str
    model_id: str
    memory_type: str
    content: str
    importance: float = 0.5
    confidence: float = 1.0
    source: str = "self"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    version: int = 1
    memory_refs: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "MemoryRecord":
        return cls(**d)


class MemoryManager:
    """
    Memory manager for stateful model reasoning.
    
    Manages:
    - working memory
    - task memory
    - episodic memory
    - peer memory
    - long-term memory
    """
    
    def __init__(
        self,
        model_id: str,
        task_id: str,
        session_id: str,
        max_working_memory: int = 50,
        max_task_memory: int = 200,
        max_episodic_memory: int = 500,
        max_peer_memory: int = 1000,
        max_long_term_memory: int = 10000,
    ):
        self.model_id = model_id
        self.task_id = task_id
        self.session_id = session_id
        self.max_working_memory = max_working_memory
        self.max_task_memory = max_task_memory
        self.max_episodic_memory = max_episodic_memory
        self.max_peer_memory = max_peer_memory
        self.max_long_term_memory = max_long_term_memory
        
        self._memories: Dict[str, MemoryRecord] = {}
        self._indices: Dict[str, List[str]] = {
            MemoryType.WORKING.value: [],
            MemoryType.TASK.value: [],
            MemoryType.EPISODIC.value: [],
            MemoryType.PEER.value: [],
            MemoryType.LONG_TERM.value: [],
        }
    
    def create_memory(
        self,
        memory_type: str,
        content: str,
        importance: float = 0.5,
        confidence: float = 1.0,
        source: str = "self",
        memory_refs: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MemoryRecord:
        """Create a new memory record."""
        memory_id = str(uuid.uuid4())
        
        record = MemoryRecord(
            memory_id=memory_id,
            task_id=self.task_id,
            session_id=self.session_id,
            model_id=self.model_id,
            memory_type=memory_type,
            content=content,
            importance=importance,
            confidence=confidence,
            source=source,
            memory_refs=memory_refs or [],
            tags=tags or [],
            metadata=metadata or {},
        )
        
        self._memories[memory_id] = record
        self._index_memory(memory_id, memory_type)
        
        # Enforce limits
        self._enforce_limits(memory_type)
        
        return record
    
    def _index_memory(self, memory_id: str, memory_type: str) -> None:
        """Add memory to type index."""
        if memory_type in self._indices:
            self._indices[memory_type].append(memory_id)
    
    def _enforce_limits(self, memory_type: str) -> None:
        """Enforce memory limits for a type."""
        limits = {
            MemoryType.WORKING.value: self.max_working_memory,
            MemoryType.TASK.value: self.max_task_memory,
            MemoryType.EPISODIC.value: self.max_episodic_memory,
            MemoryType.PEER.value: self.max_peer_memory,
            MemoryType.LONG_TERM.value: self.max_long_term_memory,
        }
        
        limit = limits.get(memory_type)
        if limit is None:
            return
        
        indices = self._indices[memory_type]
        while len(indices) > limit:
            oldest_id = indices.pop(0)
            if oldest_id in self._memories:
                del self._memories[oldest_id]
    
    def retrieve(
        self,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_importance: float = 0.0,
        limit: int = 50,
    ) -> List[MemoryRecord]:
        """Retrieve memories matching criteria."""
        candidates = []
        
        if memory_type:
            indices = self._indices.get(memory_type, [])
            candidates = [self._memories[mid] for mid in indices if mid in self._memories]
        else:
            candidates = list(self._memories.values())
        
        # Filter by tags
        if tags:
            tag_set = set(tags)
            candidates = [m for m in candidates if tag_set & set(m.tags)]
        
        # Filter by importance
        candidates = [m for m in candidates if m.importance >= min_importance]
        
        # Sort by importance descending, then timestamp
        candidates.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        
        return candidates[:limit]
    
    def update_memory(self, memory_id: str, **updates) -> Optional[MemoryRecord]:
        """Update a memory record."""
        if memory_id not in self._memories:
            return None
        
        record = self._memories[memory_id]
        for key, value in updates.items():
            if hasattr(record, key):
                setattr(record, key, value)
        
        record.version += 1
        return record
    
    def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory (move to long-term)."""
        if memory_id not in self._memories:
            return False
        
        record = self._memories[memory_id]
        old_type = record.memory_type
        
        # Remove from old index
        if old_type in self._indices and memory_id in self._indices[old_type]:
            self._indices[old_type].remove(memory_id)
        
        # Update and reindex
        record.memory_type = MemoryType.LONG_TERM.value
        self._index_memory(memory_id, MemoryType.LONG_TERM.value)
        
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory."""
        if memory_id not in self._memories:
            return False
        
        record = self._memories[memory_id]
        memory_type = record.memory_type
        
        if memory_type in self._indices and memory_id in self._indices[memory_type]:
            self._indices[memory_type].remove(memory_id)
        
        del self._memories[memory_id]
        return True
    
    def clear_type(self, memory_type: str) -> int:
        """Clear all memories of a type."""
        if memory_type not in self._indices:
            return 0
        
        count = len(self._indices[memory_type])
        for memory_id in self._indices[memory_type]:
            if memory_id in self._memories:
                del self._memories[memory_id]
        self._indices[memory_type] = []
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "total": len(self._memories),
            "by_type": {t: len(indices) for t, indices in self._indices.items()},
            "model_id": self.model_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
        }
    
    def get_working_memory_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get working memory summary for context."""
        memories = self.retrieve(memory_type=MemoryType.WORKING.value, limit=limit)
        return [
            {
                "memory_id": m.memory_id,
                "content": m.content,
                "importance": m.importance,
                "source": m.source,
                "timestamp": m.timestamp,
            }
            for m in memories
        ]
    
    def get_task_memory_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get task memory summary."""
        memories = self.retrieve(memory_type=MemoryType.TASK.value, limit=limit)
        return [
            {
                "memory_id": m.memory_id,
                "content": m.content,
                "importance": m.importance,
                "confidence": m.confidence,
                "version": m.version,
            }
            for m in memories
        ]
    
    def get_peer_memory_summary(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get peer memory summary."""
        memories = self.retrieve(memory_type=MemoryType.PEER.value, limit=limit)
        return [
            {
                "memory_id": m.memory_id,
                "content": m.content,
                "source": m.source,
                "confidence": m.confidence,
                "timestamp": m.timestamp,
            }
            for m in memories
        ]
    
    def compress_memories(self, memory_type: str, target_ratio: float = 0.5) -> int:
        """Compress memories by merging low-importance ones."""
        if memory_type not in self._indices:
            return 0
        
        indices = self._indices[memory_type]
        if len(indices) <= 1:
            return 0
        
        memories = [self._memories[mid] for mid in indices if mid in self._memories]
        memories.sort(key=lambda m: m.importance)
        
        # Remove lowest importance memories up to target ratio
        target_count = int(len(memories) * target_ratio)
        removed = 0
        
        for memory in memories[:target_count]:
            if self.delete_memory(memory.memory_id):
                removed += 1
        
        return removed
    
    def serialize(self) -> Dict[str, Any]:
        """Serialize memory state."""
        return {
            "model_id": self.model_id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "memories": [m.to_dict() for m in self._memories.values()],
            "stats": self.get_stats(),
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> "MemoryManager":
        """Deserialize memory state."""
        manager = cls(
            model_id=data["model_id"],
            task_id=data["task_id"],
            session_id=data["session_id"],
        )
        
        for memory_data in data.get("memories", []):
            record = MemoryRecord.from_dict(memory_data)
            manager._memories[record.memory_id] = record
            manager._index_memory(record.memory_id, record.memory_type)
        
        return manager
