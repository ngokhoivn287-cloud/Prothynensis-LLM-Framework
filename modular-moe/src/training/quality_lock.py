"""Quality lock system for Prothynesis training.

Enforces quality retention gates:
- Global quality retention >= 96.5%
- Per-capability quality retention >= 96.5%
- Rejects optimizations that degrade quality below threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class QualityTier(Enum):
    """Quality retention tiers."""
    REJECTED = "rejected"
    MINIMUM = "minimum"
    TARGET = "target"
    EXCELLENT = "excellent"


@dataclass
class CapabilityQuality:
    """Quality metrics for a single capability."""
    capability: str
    baseline_score: float = 0.0
    current_score: float = 0.0
    retention: float = 1.0
    threshold: float = 0.965
    
    def compute_retention(self) -> float:
        if self.baseline_score > 0:
            self.retention = self.current_score / self.baseline_score
        else:
            self.retention = 1.0 if self.current_score >= 0 else 0.0
        return self.retention
    
    def passes(self) -> bool:
        return self.compute_retention() >= self.threshold
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability": self.capability,
            "baseline_score": self.baseline_score,
            "current_score": self.current_score,
            "retention": self.retention,
            "threshold": self.threshold,
            "passes": self.passes(),
        }


@dataclass
class QualityProfile:
    """Complete quality profile for a model or population."""
    model_id: str = ""
    capabilities: Dict[str, CapabilityQuality] = field(default_factory=dict)
    global_baseline: float = 100.0
    global_current: float = 100.0
    global_threshold: float = 0.965
    global_target: float = 0.98
    notes: List[str] = field(default_factory=list)
    
    def add_capability(self, capability: str, baseline: float, current: float, threshold: Optional[float] = None) -> CapabilityQuality:
        cap = CapabilityQuality(
            capability=capability,
            baseline_score=baseline,
            current_score=current,
            threshold=threshold or self.global_threshold,
        )
        cap.compute_retention()
        self.capabilities[capability] = cap
        return cap
    
    def get_capability(self, capability: str) -> Optional[CapabilityQuality]:
        return self.capabilities.get(capability)
    
    def compute_global_retention(self) -> float:
        if self.global_baseline > 0:
            return self.global_current / self.global_baseline
        return 1.0
    
    def passes_global(self) -> bool:
        return self.compute_global_retention() >= self.global_threshold
    
    def passes_per_capability(self) -> bool:
        return all(cap.passes() for cap in self.capabilities.values())
    
    def passes_all(self) -> bool:
        return self.passes_global() and self.passes_per_capability()
    
    def get_failing_capabilities(self) -> List[str]:
        return [cap.capability for cap in self.capabilities.values() if not cap.passes()]
    
    def get_quality_tier(self) -> QualityTier:
        retention = self.compute_global_retention()
        if retention < self.global_threshold:
            return QualityTier.REJECTED
        elif retention < self.global_target:
            return QualityTier.MINIMUM
        elif retention < 0.99:
            return QualityTier.TARGET
        else:
            return QualityTier.EXCELLENT
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "global_baseline": self.global_baseline,
            "global_current": self.global_current,
            "global_retention": self.compute_global_retention(),
            "global_passes": self.passes_global(),
            "quality_tier": self.get_quality_tier().value,
            "capabilities": {name: cap.to_dict() for name, cap in self.capabilities.items()},
            "failing_capabilities": self.get_failing_capabilities(),
            "notes": self.notes,
        }


class QualityLock:
    """
    Enforces quality retention gates.
    
    Any optimization, training shortcut, or knowledge reuse must pass:
    - Global quality retention >= 96.5%
    - Per-capability quality retention >= 96.5%
    """
    
    DEFAULT_CAPABILITIES = [
        "text_generation",
        "knowledge",
        "reasoning",
        "mathematics",
        "coding",
        "verification",
        "memory",
        "collaboration",
        "tool_use",
        "long_horizon",
    ]
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.global_threshold = self.config.get("global_threshold", 0.965)
        self.global_target = self.config.get("global_target", 0.98)
        self.per_capability_threshold = self.config.get("per_capability_threshold", 0.965)
        self.capabilities = self.config.get("capabilities", self.DEFAULT_CAPABILITIES)
    
    def create_profile(self, model_id: str, baseline_scores: Dict[str, float], current_scores: Dict[str, float]) -> QualityProfile:
        """Create a quality profile from baseline and current scores."""
        profile = QualityProfile(
            model_id=model_id,
            global_baseline=100.0,
            global_current=100.0,
            global_threshold=self.global_threshold,
            global_target=self.global_target,
        )
        
        for capability in self.capabilities:
            baseline = baseline_scores.get(capability, 0.0)
            current = current_scores.get(capability, 0.0)
            profile.add_capability(capability, baseline, current, self.per_capability_threshold)
        
        # Global scores as average of capabilities
        if profile.capabilities:
            profile.global_baseline = sum(c.baseline_score for c in profile.capabilities.values()) / len(profile.capabilities)
            profile.global_current = sum(c.current_score for c in profile.capabilities.values()) / len(profile.capabilities)
        
        return profile
    
    def validate_optimization(self, profile: QualityProfile) -> Dict[str, Any]:
        """Validate whether an optimization passes quality gates."""
        passes = profile.passes_all()
        tier = profile.get_quality_tier()
        
        result = {
            "passes": passes,
            "tier": tier.value,
            "global_retention": profile.compute_global_retention(),
            "failing_capabilities": profile.get_failing_capabilities(),
            "notes": [],
        }
        
        if tier == QualityTier.REJECTED:
            result["notes"].append("Optimization rejected: quality retention below minimum threshold")
        elif tier == QualityTier.MINIMUM:
            result["notes"].append("Optimization passes minimum threshold but below target")
        elif tier == QualityTier.TARGET:
            result["notes"].append("Optimization passes target threshold")
        elif tier == QualityTier.EXCELLENT:
            result["notes"].append("Optimization passes excellent threshold")
        
        return result
    
    def validate_comparison(self, baseline_scores: Dict[str, float], optimized_scores: Dict[str, float], model_id: str = "unknown") -> Dict[str, Any]:
        """Convenience method to validate a baseline vs optimized comparison."""
        profile = self.create_profile(model_id, baseline_scores, optimized_scores)
        return self.validate_optimization(profile)
    
    def check_catastrophic_forgetting(self, before_scores: Dict[str, float], after_scores: Dict[str, float], threshold: Optional[float] = None) -> Dict[str, Any]:
        """Check if training caused catastrophic forgetting."""
        threshold = threshold or self.per_capability_threshold
        degradations = []
        
        for capability in self.capabilities:
            before = before_scores.get(capability, 0.0)
            after = after_scores.get(capability, 0.0)
            if before > 0:
                retention = after / before
                if retention < threshold:
                    degradations.append({
                        "capability": capability,
                        "before": before,
                        "after": after,
                        "retention": retention,
                    })
        
        return {
            "catastrophic_forgetting": len(degradations) > 0,
            "degraded_capabilities": degradations,
            "threshold": threshold,
        }
    
    def recommend_remediation(self, profile: QualityProfile) -> List[str]:
        """Recommend remediation for failing capabilities."""
        recommendations = []
        for capability in profile.get_failing_capabilities():
            cap = profile.capabilities[capability]
            recommendations.append(f"Increase {capability} replay/restoration for model {profile.model_id}")
        return recommendations
