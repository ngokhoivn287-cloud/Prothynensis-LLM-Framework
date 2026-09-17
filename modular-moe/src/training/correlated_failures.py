"""Correlated failure detection for Prothynensis population.

Detects when multiple models fail for the same reason,
helping identify:
- shared false assumptions
- shared evidence failures
- correlated reasoning patterns
- systemic weaknesses
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


@dataclass
class FailureEvent:
    """Single failure event from a model."""
    event_id: str
    model_id: str
    task_id: str
    failure_category: str
    description: str
    confidence: float = 0.0
    input_hash: str = ""
    output_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CorrelatedFailure:
    """Group of failures that share a common cause."""
    failure_group_id: str
    failure_category: str
    affected_models: List[str]
    affected_tasks: List[str]
    event_count: int
    first_seen: float
    last_seen: float
    shared_assumptions: List[str] = field(default_factory=list)
    shared_evidence_gaps: List[str] = field(default_factory=list)
    correlated_reasoning_patterns: List[str] = field(default_factory=list)
    severity: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class CorrelatedFailureDetector:
    """
    Detects correlated failures across the population.
    
    Identifies when multiple models fail for the same reason,
    indicating systemic issues rather than individual problems.
    """

    def __init__(self, config: dict):
        self.config = config
        self.correlation_threshold = config.get("correlated_failures", {}).get("correlation_threshold", 0.5)
        self.min_failure_count = config.get("correlated_failures", {}).get("min_failure_count", 3)
        self.time_window = config.get("correlated_failures", {}).get("time_window", 3600.0)
        self._events: List[FailureEvent] = []
        self._groups: Dict[str, CorrelatedFailure] = {}

    def record_failure(
        self,
        model_id: str,
        task_id: str,
        failure_category: str,
        description: str,
        confidence: float = 0.0,
        input_text: str = "",
        output_text: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FailureEvent:
        """Record a failure event."""
        event = FailureEvent(
            event_id=f"fail_{model_id}_{task_id}_{int(time.time()*1000)}",
            model_id=model_id,
            task_id=task_id,
            failure_category=failure_category,
            description=description,
            confidence=confidence,
            input_hash=hashlib.sha256(input_text.encode()).hexdigest()[:16] if input_text else "",
            output_hash=hashlib.sha256(output_text.encode()).hexdigest()[:16] if output_text else "",
            metadata=metadata or {},
        )
        self._events.append(event)
        self._detect_correlations(event)
        return event

    def get_correlated_failures(self, category: Optional[str] = None) -> List[CorrelatedFailure]:
        groups = list(self._groups.values())
        if category:
            groups = [g for g in groups if g.failure_category == category]
        groups.sort(key=lambda g: g.event_count, reverse=True)
        return groups

    def get_model_correlations(self, model_id: str) -> List[CorrelatedFailure]:
        return [g for g in self._groups.values() if model_id in g.affected_models]

    def get_contamination_risk(self, model_ids: List[str]) -> Dict[str, Any]:
        """Assess risk of correlated failure for a group of models."""
        relevant_groups = []
        for g in self._groups.values():
            if any(m in g.affected_models for m in model_ids):
                relevant_groups.append(g)
        if not relevant_groups:
            return {"risk": "low", "correlated_groups": 0, "affected_categories": []}
        categories = list(set(g.failure_category for g in relevant_groups))
        max_severity = max(
            (g.severity for g in relevant_groups),
            key=lambda s: {"low": 0, "medium": 1, "high": 2}.get(s, 1),
        )
        return {
            "risk": max_severity,
            "correlated_groups": len(relevant_groups),
            "affected_categories": categories,
            "recommended_actions": [
                "recruit_contrarian_models",
                "recruit_verifier_models",
                "increase_verification_rounds",
            ],
        }

    def _detect_correlations(self, new_event: FailureEvent) -> None:
        now = time.time()
        recent = [e for e in self._events if now - e.timestamp <= self.time_window]
        category_events = [e for e in recent if e.failure_category == new_event.failure_category]
        if len(category_events) < self.min_failure_count:
            return
        affected_models = list(set(e.model_id for e in category_events))
        affected_tasks = list(set(e.task_id for e in category_events))
        group_id = hashlib.sha256(
            f"{new_event.failure_category}_{'-'.join(sorted(affected_models))}".encode()
        ).hexdigest()[:16]
        if group_id not in self._groups:
            self._groups[group_id] = CorrelatedFailure(
                failure_group_id=group_id,
                failure_category=new_event.failure_category,
                affected_models=affected_models,
                affected_tasks=affected_tasks,
                event_count=len(category_events),
                first_seen=min(e.timestamp for e in category_events),
                last_seen=max(e.timestamp for e in category_events),
                severity="high" if len(affected_models) >= 5 else "medium",
            )
        else:
            group = self._groups[group_id]
            group.event_count = len(category_events)
            group.last_seen = now
            group.affected_models = list(set(group.affected_models + affected_models))
            group.affected_tasks = list(set(group.affected_tasks + affected_tasks))
            if len(group.affected_models) >= 5:
                group.severity = "high"

    def get_statistics(self) -> Dict[str, Any]:
        if not self._events:
            return {"total_events": 0, "correlated_groups": 0}
        category_counts: Dict[str, int] = defaultdict(int)
        for e in self._events:
            category_counts[e.failure_category] += 1
        return {
            "total_events": len(self._events),
            "correlated_groups": len(self._groups),
            "category_distribution": dict(category_counts),
            "high_severity_groups": sum(1 for g in self._groups.values() if g.severity == "high"),
        }

    def to_dict(self) -> dict:
        return {
            "events": [e.to_dict() for e in self._events[-1000:]],
            "groups": [g.to_dict() for g in self._groups.values()],
            "statistics": self.get_statistics(),
        }
