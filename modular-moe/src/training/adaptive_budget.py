"""Adaptive training budget allocation for Prothynesis.

Allocates compute, tokens, and wall-time budgets per model based on:
- current quality
- marginal quality gain
- remaining time
- model priority
- population contribution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import time


class BudgetPhase(Enum):
    """Training budget phases."""
    WARMUP = "warmup"
    ADAPTIVE = "adaptive"
    FINAL = "final"
    COMPLETE = "complete"
    STOPPED = "stopped"


class StopReason(Enum):
    """Reasons for stopping training."""
    QUALITY_TARGET_REACHED = "quality_target_reached"
    VALIDATION_PLATEAU = "validation_plateau"
    CAPABILITY_PLATEAU = "capability_plateau"
    MARGINAL_GAIN_TOO_LOW = "marginal_gain_too_low"
    DEADLINE_REACHED = "deadline_reached"
    COMPUTE_BUDGET_EXHAUSTED = "compute_budget_exhausted"
    OPTIMIZATION_REJECTED = "optimization_rejected"


@dataclass
class TrainingBudget:
    """Training budget for a single model."""
    model_id: str
    max_wall_time_seconds: float
    max_tokens: int
    max_compute_flops: float
    quality_target: float
    priority: float = 1.0
    deadline_timestamp: Optional[float] = None
    
    # Tracking
    wall_time_seconds: float = 0.0
    tokens_seen: int = 0
    compute_used: float = 0.0
    current_quality: float = 0.0
    marginal_gain: float = 0.0
    phase: str = BudgetPhase.WARMUP.value
    
    # Progress
    start_time: float = field(default_factory=time.time)
    last_evaluation_time: float = field(default_factory=time.time)
    last_quality: float = 0.0
    quality_plateau_count: int = 0
    
    def remaining_wall_time(self) -> float:
        if self.deadline_timestamp is not None:
            return max(0.0, self.deadline_timestamp - time.time())
        return max(0.0, self.max_wall_time_seconds - self.wall_time_seconds)
    
    def remaining_tokens(self) -> int:
        return max(0, self.max_tokens - self.tokens_seen)
    
    def remaining_compute(self) -> float:
        return max(0.0, self.max_compute_flops - self.compute_used)
    
    def progress_fraction(self) -> float:
        time_progress = min(1.0, self.wall_time_seconds / self.max_wall_time_seconds) if self.max_wall_time_seconds > 0 else 0.0
        token_progress = min(1.0, self.tokens_seen / self.max_tokens) if self.max_tokens > 0 else 0.0
        return max(time_progress, token_progress)
    
    def should_stop(self, marginal_gain_threshold: float = 0.001, plateau_patience: int = 3) -> Optional[StopReason]:
        """Determine if training should stop."""
        if self.current_quality >= self.quality_target:
            return StopReason.QUALITY_TARGET_REACHED
        
        if self.remaining_wall_time() <= 0:
            return StopReason.DEADLINE_REACHED
        
        if self.remaining_tokens() <= 0:
            return StopReason.COMPUTE_BUDGET_EXHAUSTED
        
        if self.remaining_compute() <= 0:
            return StopReason.COMPUTE_BUDGET_EXHAUSTED
        
        if self.marginal_gain < marginal_gain_threshold and self.phase == BudgetPhase.FINAL.value:
            self.quality_plateau_count += 1
            if self.quality_plateau_count >= plateau_patience:
                return StopReason.MARGINAL_GAIN_TOO_LOW
        else:
            self.quality_plateau_count = 0
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_wall_time_seconds": self.max_wall_time_seconds,
            "max_tokens": self.max_tokens,
            "max_compute_flops": self.max_compute_flops,
            "quality_target": self.quality_target,
            "priority": self.priority,
            "wall_time_seconds": self.wall_time_seconds,
            "tokens_seen": self.tokens_seen,
            "compute_used": self.compute_used,
            "current_quality": self.current_quality,
            "marginal_gain": self.marginal_gain,
            "phase": self.phase,
            "remaining_wall_time": self.remaining_wall_time(),
            "remaining_tokens": self.remaining_tokens(),
            "progress_fraction": self.progress_fraction(),
        }


@dataclass
class BudgetAllocation:
    """Budget allocation for a set of models."""
    model_budgets: Dict[str, TrainingBudget]
    total_wall_time: float
    total_tokens: int
    total_compute: float
    allocation_timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_wall_time": self.total_wall_time,
            "total_tokens": self.total_tokens,
            "total_compute": self.total_compute,
            "allocation_timestamp": self.allocation_timestamp,
            "model_budgets": {mid: budget.to_dict() for mid, budget in self.model_budgets.items()},
        }


class AdaptiveBudgetAllocator:
    """
    Allocates training budgets to models based on their value and progress.
    
    Uses remaining training value (RTV) heuristic:
    RTV = Expected Remaining Quality Gain / Remaining Time
    
    Prioritizes models with highest RTV.
    
    Restart policy:
    - Production default requires tokens_seen >= default_token_budget * 0.5
      before a restart is considered, to avoid restarting unstable models.
    - Stress tests and early-stage experiments may lower this fraction via
      ``restart_token_fraction``.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.default_token_budget = config.get("default_token_budget", 1_000_000_000)
        self.default_wall_time_budget = config.get("default_wall_time_budget", 86400 * 30)  # 30 days
        self.marginal_gain_threshold = config.get("marginal_gain_threshold", 0.001)
        self.plateau_patience = config.get("plateau_patience", 3)
        self.restart_token_fraction = config.get("restart_token_fraction", 0.5)
    
    def allocate_initial_budget(self, model_profiles: List[Dict[str, Any]]) -> BudgetAllocation:
        """Allocate initial budgets for a set of models."""
        model_budgets = {}
        
        for profile in model_profiles:
            model_id = profile["model_id"]
            priority = profile.get("priority", 1.0)
            deadline_days = profile.get("deadline_days", 30)
            
            wall_time = min(self.default_wall_time_budget * priority, deadline_days * 86400)
            tokens = int(self.default_token_budget * priority)
            
            budget = TrainingBudget(
                model_id=model_id,
                max_wall_time_seconds=wall_time,
                max_tokens=tokens,
                max_compute_flops=tokens * 1e-9,  # rough estimate
                quality_target=profile.get("quality_target", 0.9),
                priority=priority,
                deadline_timestamp=time.time() + wall_time,
            )
            model_budgets[model_id] = budget
        
        total_wall = sum(b.max_wall_time_seconds for b in model_budgets.values())
        total_tokens = sum(b.max_tokens for b in model_budgets.values())
        total_compute = sum(b.max_compute_flops for b in model_budgets.values())
        
        return BudgetAllocation(
            model_budgets=model_budgets,
            total_wall_time=total_wall,
            total_tokens=total_tokens,
            total_compute=total_compute,
        )
    
    def reallocate_budget(self, allocation: BudgetAllocation, quality_updates: Dict[str, float], marginal_gains: Dict[str, float]) -> BudgetAllocation:
        """Reallocate budgets based on current progress and marginal gains."""
        for model_id, budget in allocation.model_budgets.items():
            quality = quality_updates.get(model_id, budget.current_quality)
            marginal_gain = marginal_gains.get(model_id, budget.marginal_gain)
            
            budget.current_quality = quality
            budget.marginal_gain = marginal_gain
            
            # Phase transitions
            if budget.phase == BudgetPhase.WARMUP.value and budget.tokens_seen > self.default_token_budget * 0.1:
                budget.phase = BudgetPhase.ADAPTIVE.value
            elif budget.phase == BudgetPhase.ADAPTIVE.value and quality >= budget.quality_target:
                budget.phase = BudgetPhase.FINAL.value
        
        return allocation
    
    def compute_rtv(self, budget: TrainingBudget) -> float:
        """Compute remaining training value heuristic."""
        remaining_quality = max(0.0, budget.quality_target - budget.current_quality)
        remaining_time = budget.remaining_wall_time()
        
        if remaining_time <= 0:
            return 0.0
        
        rtv = remaining_quality / remaining_time
        return rtv * budget.priority
    
    def prioritize_models(self, allocation: BudgetAllocation) -> List[str]:
        """Return model IDs sorted by remaining training value."""
        scored = []
        for model_id, budget in allocation.model_budgets.items():
            rtv = self.compute_rtv(budget)
            scored.append((model_id, rtv))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in scored]
    
    def should_restart(self, budget: TrainingBudget, max_restarts: int = 3, restart_value_threshold: float = 0.05) -> bool:
        """Determine if a model should get a fresh restart."""
        if budget.current_quality <= 0:
            return False
        
        # Only restart if in adaptive phase and meaningful improvement is still expected
        if budget.phase != BudgetPhase.ADAPTIVE.value:
            return False
        
        # Must have trained enough to be stable.
        # Production default is 50% of the token budget; stress tests may lower
        # this via ``restart_token_fraction``.
        if budget.tokens_seen < self.default_token_budget * self.restart_token_fraction:
            return False
        
        # Must have enough remaining time
        if budget.remaining_wall_time() < self.default_wall_time_budget * 0.3:
            return False
        
        # Current quality must be decent but not at target
        if budget.current_quality < 0.3 or budget.current_quality >= budget.quality_target:
            return False
        
        return True
