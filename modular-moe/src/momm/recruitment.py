"""Dynamic recruitment and pure-token activation accounting for Prothynesis."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum


class RecruitmentStrategy(Enum):
    """Strategies for dynamic recruitment."""
    STATIC = "static"
    DIFFICULTY_BASED = "difficulty_based"
    UNCERTAINTY_BASED = "uncertainty_based"
    DISAGREEMENT_BASED = "disagreement_based"
    HYBRID = "hybrid"


@dataclass
class RecruitmentContext:
    """Context for recruitment decision."""
    task_difficulty: float = 0.5
    uncertainty: float = 0.0
    disagreement: float = 0.0
    missing_information: float = 0.0
    expected_information_gain: float = 0.0
    verification_needed: bool = False
    compute_budget: float = 1.0
    latency_budget: float = 60.0
    current_active_models: int = 0
    max_active_models: int = 32
    min_active_models: int = 1


@dataclass
class ActivationBudget:
    """Tracks pure-token activation parameters."""
    active_model_count: int = 0
    parameters_per_model: int = 0
    active_parameters: int = 0
    estimated_flops: float = 0.0
    estimated_memory_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def active_parameters_billions(self) -> float:
        return self.active_parameters / 1e9

    @property
    def active_parameters_trillions(self) -> float:
        return self.active_parameters / 1e12

    def update(self, active_models: List[Any]) -> None:
        """Update budget based on currently active models."""
        self.active_model_count = len(active_models)
        if active_models and hasattr(active_models[0], "parameters"):
            self.parameters_per_model = sum(
                p.numel() for p in active_models[0].parameters()
            )
        elif active_models and hasattr(active_models[0], "get_parameter_count"):
            self.parameters_per_model = active_models[0].get_parameter_count().get("total", 0)
        else:
            self.parameters_per_model = 0

        self.active_parameters = self.active_model_count * self.parameters_per_model
        # Rough FLOP estimate: 2 * params * tokens
        self.estimated_flops = 2.0 * self.active_parameters * 1000  # per 1k tokens
        # Rough memory estimate: 4 bytes per param for fp32
        self.estimated_memory_mb = (self.active_parameters * 4) / (1024 * 1024)
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "active_model_count": self.active_model_count,
            "parameters_per_model": self.parameters_per_model,
            "active_parameters": self.active_parameters,
            "active_parameters_billions": self.active_parameters_billions,
            "active_parameters_trillions": self.active_parameters_trillions,
            "estimated_flops": self.estimated_flops,
            "estimated_memory_mb": self.estimated_memory_mb,
            "timestamp": self.timestamp,
        }


class DynamicRecruiter:
    """
    Dynamically recruits models based on task requirements.
    
    Supports:
    - Difficulty-based recruitment
    - Uncertainty-based recruitment
    - Disagreement-based recruitment
    - Compute/latency budget enforcement
    """

    def __init__(self, config: dict):
        self.config = config
        self.strategy = RecruitmentStrategy(
            config.get("recruitment", {}).get("strategy", "hybrid")
        )
        self.min_active = config.get("recruitment", {}).get("min_active_models", 1)
        self.max_active = config.get("recruitment", {}).get("max_active_models", 32)
        self.default_top_k = config.get("recruitment", {}).get("default_top_k", 3)
        self.compute_budget = config.get("recruitment", {}).get("compute_budget", 1.0)
        self.latency_budget = config.get("recruitment", {}).get("latency_budget", 60.0)
        self.confidence_threshold = config.get("recruitment", {}).get("confidence_threshold", 0.85)
        self.disagreement_threshold = config.get("recruitment", {}).get("disagreement_threshold", 0.15)

    def recruit(
        self,
        context: RecruitmentContext,
        available_models: List[Any],
        active_models: List[Any],
    ) -> List[Any]:
        """
        Recruit models based on context.
        
        Returns the list of models that should be active.
        """
        if self.strategy == RecruitmentStrategy.STATIC:
            return self._recruit_static(available_models, active_models, context)
        elif self.strategy == RecruitmentStrategy.DIFFICULTY_BASED:
            return self._recruit_difficulty_based(available_models, active_models, context)
        elif self.strategy == RecruitmentStrategy.UNCERTAINTY_BASED:
            return self._recruit_uncertainty_based(available_models, active_models, context)
        elif self.strategy == RecruitmentStrategy.DISAGREEMENT_BASED:
            return self._recruit_disagreement_based(available_models, active_models, context)
        else:
            return self._recruit_hybrid(available_models, active_models, context)

    def _recruit_static(
        self,
        available_models: List[Any],
        active_models: List[Any],
        context: RecruitmentContext,
    ) -> List[Any]:
        """Static recruitment: use default_top_k models."""
        return available_models[: self.default_top_k]

    def _recruit_difficulty_based(
        self,
        available_models: List[Any],
        active_models: List[Any],
        context: RecruitmentContext,
    ) -> List[Any]:
        """Recruit more models for harder tasks."""
        # Scale with difficulty: easy=1-3, normal=3-10, hard=10-30, complex=30+
        if context.task_difficulty < 0.3:
            target = max(self.min_active, 3)
        elif context.task_difficulty < 0.6:
            target = max(self.min_active, int(10 * context.compute_budget))
        elif context.task_difficulty < 0.8:
            target = max(self.min_active, int(30 * context.compute_budget))
        else:
            target = min(self.max_active, int(100 * context.compute_budget))

        target = min(target, len(available_models))
        return available_models[:target]

    def _recruit_uncertainty_based(
        self,
        available_models: List[Any],
        active_models: List[Any],
        context: RecruitmentContext,
    ) -> List[Any]:
        """Recruit more models when uncertainty is high."""
        base_count = self.default_top_k
        uncertainty_factor = context.uncertainty / max(0.01, 1.0 - self.confidence_threshold)
        target = int(base_count * (1.0 + uncertainty_factor))
        target = max(self.min_active, min(self.max_active, target))
        return available_models[:target]

    def _recruit_disagreement_based(
        self,
        available_models: List[Any],
        active_models: List[Any],
        context: RecruitmentContext,
    ) -> List[Any]:
        """Recruit more models when there is disagreement."""
        if context.disagreement > self.disagreement_threshold:
            target = min(self.max_active, len(available_models))
        else:
            target = self.default_top_k
        return available_models[:target]

    def _recruit_hybrid(
        self,
        available_models: List[Any],
        active_models: List[Any],
        context: RecruitmentContext,
    ) -> List[Any]:
        """Hybrid recruitment combining multiple signals."""
        target = self.default_top_k

        # Scale up based on difficulty
        if context.task_difficulty > 0.6:
            target = max(target, int(10 * context.compute_budget))

        # Scale up based on uncertainty
        if context.uncertainty > (1.0 - self.confidence_threshold):
            target = max(target, int(target * 2.0))

        # Scale up based on disagreement
        if context.disagreement > self.disagreement_threshold:
            target = max(target, int(20 * context.compute_budget))

        # Scale up if verification needed
        if context.verification_needed:
            target = max(target, int(5 * context.compute_budget))

        # Scale up based on expected information gain
        if context.expected_information_gain > 0.3:
            target = max(target, int(target * 1.5))

        # Enforce budgets
        target = max(self.min_active, min(self.max_active, target))
        target = min(target, len(available_models))

        return available_models[:target]

    def estimate_activation(
        self,
        active_models: List[Any],
    ) -> ActivationBudget:
        """Estimate pure-token activation parameters."""
        budget = ActivationBudget()
        budget.update(active_models)
        return budget


class RecruitmentBudget:
    """Enforces compute and latency budgets for recruitment."""

    def __init__(self, config: dict):
        self.compute_budget = config.get("recruitment", {}).get("compute_budget", 1.0)
        self.latency_budget = config.get("recruitment", {}).get("latency_budget", 60.0)
        self.max_active_models = config.get("recruitment", {}).get("max_active_models", 32)

    def can_activate(self, additional_count: int, current_activation: ActivationBudget) -> bool:
        """Check if additional models can be activated within budget."""
        new_count = current_activation.active_model_count + additional_count
        if new_count > self.max_active_models:
            return False

        # Estimate additional compute
        additional_params = additional_count * current_activation.parameters_per_model
        new_total_params = current_activation.active_parameters + additional_params

        # Check compute budget (simplified: assume budget is fraction of max)
        max_params = 77_161_472 * self.max_active_models  # 75M model
        if new_total_params > max_params * self.compute_budget:
            return False

        return True

    def get_max_activatable(self, current_activation: ActivationBudget) -> int:
        """Get maximum number of models that can be activated."""
        remaining = self.max_active_models - current_activation.active_model_count
        return max(0, remaining)
