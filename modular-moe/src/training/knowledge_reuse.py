"""Knowledge reuse system for Prothynesis population training.

Supports:
- Direct experience reuse (verified data/traces)
- Distillation reuse (teacher outputs)
- Hard-example reuse
- Curriculum reuse
- Checkpoint reuse (optional)
- Provenance and contamination checks
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class KnowledgeType(Enum):
    """Types of reusable knowledge."""
    VERIFIED_SOLUTION = "verified_solution"
    HARD_EXAMPLE = "hard_example"
    SUCCESSFUL_TRACE = "successful_trace"
    VERIFICATION_TRACE = "verification_trace"
    TEACHER_EXAMPLE = "teacher_example"
    COLLABORATION_TRACE = "collaboration_trace"
    CURRICULUM_DISCOVERY = "curriculum_discovery"
    DATASET_STATISTICS = "dataset_statistics"
    FAILURE_PATTERN = "failure_pattern"
    CAPABILITY_PROFILE = "capability_profile"
    CHECKPOINT_KNOWLEDGE = "checkpoint_knowledge"


class ReusePolicy(Enum):
    """Knowledge reuse policies."""
    EXPERIENCE_FIRST = "experience_first"
    WEIGHTS_ONLY_IF_HELPFUL = "weights_only_if_helpful"
    DISABLED = "disabled"


@dataclass
class KnowledgeEntry:
    """A single knowledge entry."""
    entry_id: str
    knowledge_type: str
    source_model_id: str
    target_model_id: Optional[str] = None
    content: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0
    provenance: Dict[str, Any] = field(default_factory=dict)
    license_constraints: List[str] = field(default_factory=list)
    checksum: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def compute_checksum(self) -> str:
        import json
        content_str = json.dumps(self.content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def verify_checksum(self) -> bool:
        return self.checksum == self.compute_checksum()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "knowledge_type": self.knowledge_type,
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "content": self.content,
            "quality_score": self.quality_score,
            "provenance": self.provenance,
            "license_constraints": self.license_constraints,
            "checksum": self.checksum,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class ReusePolicyConfig:
    """Configuration for knowledge reuse."""
    enabled: bool = True
    policy: str = ReusePolicy.EXPERIENCE_FIRST.value
    weight_reuse_probability: float = 0.0
    max_generation_depth: int = 1
    min_quality_threshold: float = 0.7
    require_verification: bool = True
    contamination_check: bool = True
    diversity_test: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "policy": self.policy,
            "weight_reuse_probability": self.weight_reuse_probability,
            "max_generation_depth": self.max_generation_depth,
            "min_quality_threshold": self.min_quality_threshold,
            "require_verification": self.require_verification,
            "contamination_check": self.contamination_check,
            "diversity_test": self.diversity_test,
        }


class KnowledgeStore:
    """
    Central store for reusable knowledge/experience.
    
    Maintains:
    - Verified solutions
    - Hard examples
    - Successful traces
    - Teacher outputs
    - Curriculum discoveries
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.entries: Dict[str, KnowledgeEntry] = {}
        self.policy = ReusePolicyConfig(**config.get("reuse_policy", {}))
        self._index: Dict[str, List[str]] = {}  # type -> list of entry_ids
    
    def add_entry(self, entry: KnowledgeEntry) -> bool:
        """Add a knowledge entry if it passes quality gates."""
        if not self.policy.enabled:
            return False
        
        # Quality gate
        if entry.quality_score < self.policy.min_quality_threshold:
            return False
        
        # Verification gate
        if self.policy.require_verification and not entry.verify_checksum():
            return False
        
        # Contamination check
        if self.policy.contamination_check and self._is_contaminated(entry):
            return False
        
        entry.checksum = entry.compute_checksum()
        self.entries[entry.entry_id] = entry
        
        # Update index
        ktype = entry.knowledge_type
        if ktype not in self._index:
            self._index[ktype] = []
        self._index[ktype].append(entry.entry_id)
        
        return True
    
    def _is_contaminated(self, entry: KnowledgeEntry) -> bool:
        """Check if entry is contaminated (e.g., contains benchmark answers)."""
        metadata = entry.metadata or {}
        return metadata.get("is_benchmark", False) or metadata.get("is_validation", False)
    
    def get_entries(self, knowledge_type: Optional[str] = None, min_quality: float = 0.0, source_model_id: Optional[str] = None) -> List[KnowledgeEntry]:
        """Get knowledge entries matching criteria."""
        results = []
        entry_ids = self._index.get(knowledge_type, list(self.entries.keys())) if knowledge_type else list(self.entries.keys())
        
        for entry_id in entry_ids:
            entry = self.entries.get(entry_id)
            if entry is None:
                continue
            if entry.quality_score < min_quality:
                continue
            if source_model_id is not None and entry.source_model_id != source_model_id:
                continue
            results.append(entry)
        
        return results
    
    def promote_to_population(self, entry: KnowledgeEntry, target_generation: int) -> bool:
        """Promote knowledge to population training corpus."""
        entry.metadata["promoted_to_generation"] = target_generation
        entry.metadata["promoted_at"] = time.time()
        return self.add_entry(entry)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge store statistics."""
        type_counts: Dict[str, int] = {}
        for entry in self.entries.values():
            type_counts[entry.knowledge_type] = type_counts.get(entry.knowledge_type, 0) + 1
        
        return {
            "total_entries": len(self.entries),
            "type_distribution": type_counts,
            "policy": self.policy.to_dict(),
        }


class KnowledgeReuseManager:
    """
    Manages knowledge reuse for model training.
    
    Default policy: experience first, weights only when shown to help.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.store = KnowledgeStore(config)
        self.policy = ReusePolicyConfig(**config.get("reuse_policy", {}))
    
    def prepare_training_knowledge(self, model_id: str, target_generation: int, knowledge_types: Optional[List[str]] = None) -> List[KnowledgeEntry]:
        """Prepare knowledge for a model's training."""
        if not self.policy.enabled:
            return []
        
        entries = []
        for ktype in knowledge_types or [kt.value for kt in KnowledgeType]:
            entries.extend(self.store.get_entries(knowledge_type=ktype, min_quality=self.policy.min_quality_threshold))
        
        return entries
    
    def record_knowledge(self, model_id: str, knowledge_type: str, content: Dict[str, Any], quality_score: float, provenance: Optional[Dict[str, Any]] = None) -> Optional[KnowledgeEntry]:
        """Record new knowledge from a model."""
        entry = KnowledgeEntry(
            entry_id=f"{model_id}_{knowledge_type}_{int(time.time() * 1000)}",
            knowledge_type=knowledge_type,
            source_model_id=model_id,
            target_model_id=None,
            content=content,
            quality_score=quality_score,
            provenance=provenance or {"source": model_id, "timestamp": time.time()},
        )
        
        if self.store.add_entry(entry):
            return entry
        return None
    
    def should_reuse_weights(self, source_model_id: str, target_model_id: str) -> bool:
        """Determine if weight reuse should be used."""
        if not self.policy.enabled:
            return False
        
        if self.policy.policy == ReusePolicy.DISABLED.value:
            return False
        
        if self.policy.policy == ReusePolicy.EXPERIENCE_FIRST.value:
            import random
            return random.random() < self.policy.weight_reuse_probability
        
        return False
    
    def get_reuse_stats(self) -> Dict[str, Any]:
        """Get knowledge reuse statistics."""
        return self.store.get_stats()
