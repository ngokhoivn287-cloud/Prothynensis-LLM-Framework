"""Hard-example mining and failure logging for Prothynesis."""

from __future__ import annotations

import json
import time
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum


class FailureCategory(Enum):
    """Categories of model failure."""
    FACTUAL_ERROR = "factual_error"
    REASONING_ERROR = "reasoning_error"
    LOGIC_ERROR = "logic_error"
    MATH_ERROR = "math_error"
    CODING_ERROR = "coding_error"
    MISSING_CONSTRAINT = "missing_constraint"
    HALLUCINATION = "hallucination"
    VERIFICATION_FAILURE = "verification_failure"
    CONTEXT_FAILURE = "context_failure"
    MEMORY_FAILURE = "memory_failure"
    COLLABORATION_FAILURE = "collaboration_failure"
    TOOL_USE_FAILURE = "tool_use_failure"
    PLANNING_FAILURE = "planning_failure"


class DifficultyLevel(Enum):
    """Estimated difficulty of a hard example."""
    D0 = "D0"
    D1 = "D1"
    D2 = "D2"
    D3 = "D3"
    D4 = "D4"
    D5 = "D5"


@dataclass
class HardExample:
    """A hard example mined from population failure."""
    example_id: str
    task_id: str
    prompt: str
    wrong_candidates: List[str]
    correct_candidate: Optional[str]
    critiques: List[str]
    verifications: List[str]
    failure_categories: List[str]
    difficulty: str
    domain: str
    task_type: str
    model_ids: List[str]
    confidence_scores: List[float]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "HardExample":
        return cls(**d)


@dataclass
class FailureLog:
    """Log entry for a model failure."""
    failure_id: str
    model_id: str
    task_id: str
    category: str
    description: str
    input_ids: Optional[List[int]] = None
    output_ids: Optional[List[int]] = None
    loss: Optional[float] = None
    confidence: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class HardExampleMiner:
    """
    Mines hard examples from population failures.
    
    Flow:
    1. Detect task where population fails or disagrees
    2. Record wrong candidates, critiques, verifications
    3. Classify failure category
    4. Estimate difficulty
    5. Store for curriculum update
    """

    def __init__(self, config: dict):
        self.config = config
        self.disagreement_threshold = config.get("hard_example_mining", {}).get("disagreement_threshold", 0.3)
        self.confidence_threshold = config.get("hard_example_mining", {}).get("confidence_threshold", 0.5)
        self.min_failure_count = config.get("hard_example_mining", {}).get("min_failure_count", 2)
        self.hard_examples: List[HardExample] = []
        self.failure_logs: List[FailureLog] = []

    def mine_from_population(
        self,
        task_id: str,
        prompt: str,
        candidates: Dict[str, Any],
        verifications: Dict[str, Any],
    ) -> Optional[HardExample]:
        """
        Mine a hard example from population outputs.
        
        Args:
            task_id: Unique task identifier
            prompt: Task prompt
            candidates: Dict of model_id -> candidate output
            verifications: Dict of model_id -> verification result
            
        Returns:
            HardExample if task is hard, else None
        """
        # Check if task is hard (high disagreement or low confidence)
        confidences = [v.get("confidence", 0.5) for v in verifications.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5

        # Count failures
        failures = sum(1 for v in verifications.values() if not v.get("passed", True))

        if failures < self.min_failure_count and avg_confidence > self.confidence_threshold:
            return None

        # Extract wrong and correct candidates
        wrong_candidates = []
        correct_candidate = None
        critiques = []
        verification_results = []
        model_ids = []
        confidence_scores = []

        for model_id, candidate in candidates.items():
            model_ids.append(model_id)
            confidence_scores.append(verifications.get(model_id, {}).get("confidence", 0.5))

            if verifications.get(model_id, {}).get("passed", False):
                correct_candidate = candidate.get("text", "")
            else:
                wrong_candidates.append(candidate.get("text", ""))

            if "critique" in candidate:
                critiques.append(candidate["critique"])
            verification_results.append(verifications.get(model_id, {}).get("result", ""))

        # Classify failure
        failure_categories = self._classify_failure(
            prompt=prompt,
            candidates=candidates,
            verifications=verifications,
        )

        # Estimate difficulty
        difficulty = self._estimate_difficulty(
            avg_confidence=avg_confidence,
            failure_count=failures,
            disagreement=len(set(c.get("text", "") for c in candidates.values())),
        )

        example = HardExample(
            example_id=f"hard_{int(time.time() * 1000)}",
            task_id=task_id,
            prompt=prompt,
            wrong_candidates=wrong_candidates,
            correct_candidate=correct_candidate,
            critiques=critiques,
            verifications=verification_results,
            failure_categories=failure_categories,
            difficulty=difficulty,
            domain=self._classify_domain(prompt),
            task_type=self._classify_task_type(prompt),
            model_ids=model_ids,
            confidence_scores=confidence_scores,
        )

        self.hard_examples.append(example)
        return example

    def _classify_failure(
        self,
        prompt: str,
        candidates: Dict[str, Any],
        verifications: Dict[str, Any],
    ) -> List[str]:
        """Classify failure category from task and results."""
        categories = []

        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["calculate", "compute", "solve", "equation"]):
            categories.append(FailureCategory.MATH_ERROR.value)
        if any(kw in prompt_lower for kw in ["code", "function", "program", "debug"]):
            categories.append(FailureCategory.CODING_ERROR.value)
        if any(kw in prompt_lower for kw in ["prove", "proof", "theorem"]):
            categories.append(FailureCategory.REASONING_ERROR.value)
        if any(kw in prompt_lower for kw in ["fact", "true", "false", "correct"]):
            categories.append(FailureCategory.FACTUAL_ERROR.value)

        # Check for hallucination (high confidence but wrong)
        for model_id, v in verifications.items():
            if v.get("confidence", 0) > 0.8 and not v.get("passed", True):
                categories.append(FailureCategory.HALLUCINATION.value)
                break

        # Check for verification failure
        if any(not v.get("passed", True) for v in verifications.values()):
            categories.append(FailureCategory.VERIFICATION_FAILURE.value)

        return categories or [FailureCategory.REASONING_ERROR.value]

    def _estimate_difficulty(
        self,
        avg_confidence: float,
        failure_count: int,
        disagreement: int,
    ) -> str:
        """Estimate difficulty level."""
        score = 0
        if avg_confidence < 0.3:
            score += 2
        elif avg_confidence < 0.5:
            score += 1

        score += min(failure_count, 3)
        score += min(disagreement, 3)

        if score >= 5:
            return DifficultyLevel.D5.value
        elif score >= 4:
            return DifficultyLevel.D4.value
        elif score >= 3:
            return DifficultyLevel.D3.value
        elif score >= 2:
            return DifficultyLevel.D2.value
        elif score >= 1:
            return DifficultyLevel.D1.value
        else:
            return DifficultyLevel.D0.value

    def _classify_domain(self, prompt: str) -> str:
        """Classify domain from prompt."""
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["math", "calculate", "equation"]):
            return "mathematics"
        elif any(kw in prompt_lower for kw in ["code", "function", "program"]):
            return "programming"
        elif any(kw in prompt_lower for kw in ["science", "physics", "chemistry"]):
            return "science"
        elif any(kw in prompt_lower for kw in ["prove", "logic", "reason"]):
            return "reasoning"
        else:
            return "general"

    def _classify_task_type(self, prompt: str) -> str:
        """Classify task type from prompt."""
        prompt_lower = prompt.lower()
        if any(kw in prompt_lower for kw in ["verify", "check", "correct"]):
            return "verification"
        elif any(kw in prompt_lower for kw in ["explain", "why"]):
            return "explanation"
        elif any(kw in prompt_lower for kw in ["solve", "compute"]):
            return "problem_solving"
        else:
            return "reasoning"

    def log_failure(
        self,
        model_id: str,
        task_id: str,
        category: str,
        description: str,
        loss: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> FailureLog:
        """Log a model failure."""
        failure = FailureLog(
            failure_id=f"fail_{int(time.time() * 1000)}_{model_id}",
            model_id=model_id,
            task_id=task_id,
            category=category,
            description=description,
            loss=loss,
            confidence=confidence,
        )
        self.failure_logs.append(failure)
        return failure

    def get_hard_examples(self, domain: Optional[str] = None, difficulty: Optional[str] = None) -> List[HardExample]:
        """Get filtered hard examples."""
        examples = self.hard_examples
        if domain:
            examples = [e for e in examples if e.domain == domain]
        if difficulty:
            examples = [e for e in examples if e.difficulty == difficulty]
        return examples

    def get_failure_stats(self) -> Dict[str, Any]:
        """Get failure statistics."""
        if not self.failure_logs:
            return {"total_failures": 0}

        category_counts: Dict[str, int] = {}
        model_counts: Dict[str, int] = {}
        for log in self.failure_logs:
            category_counts[log.category] = category_counts.get(log.category, 0) + 1
            model_counts[log.model_id] = model_counts.get(log.model_id, 0) + 1

        return {
            "total_failures": len(self.failure_logs),
            "total_hard_examples": len(self.hard_examples),
            "category_distribution": category_counts,
            "model_failure_counts": model_counts,
        }

    def save(self, output_path: str) -> None:
        """Save hard examples and failure logs."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "hard_examples": [e.to_dict() for e in self.hard_examples],
            "failure_logs": [f.to_dict() for f in self.failure_logs],
            "stats": self.get_failure_stats(),
        }

        with open(output, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, input_path: str) -> None:
        """Load hard examples and failure logs."""
        with open(input_path, "r") as f:
            data = json.load(f)

        self.hard_examples = [HardExample.from_dict(e) for e in data.get("hard_examples", [])]
        self.failure_logs = [FailureLog(**f) for f in data.get("failure_logs", [])]


class CurriculumUpdater:
    """
    Updates dataset mixtures based on hard examples and failures.
    
    Uses hard-example statistics to:
    - Increase weight on difficult domains
    - Increase difficulty exposure for struggling models
    - Add hard examples to training pool
    """

    def __init__(self, config: dict):
        self.config = config
        self.mining = HardExampleMiner(config)

    def update_mixture(
        self,
        mixture: "ModelMixture",  # noqa: F821
        hard_examples: List[HardExample],
        failure_stats: Dict[str, Any],
    ) -> "ModelMixture":
        """
        Update model mixture based on hard examples.
        
        Args:
            mixture: Current model mixture
            hard_examples: Mined hard examples
            failure_stats: Failure statistics
            
        Returns:
            Updated mixture
        """
        if not hard_examples:
            return mixture

        # Boost domains where failures occurred
        domain_failures: Dict[str, int] = {}
        for example in hard_examples:
            domain_failures[example.domain] = domain_failures.get(example.domain, 0) + 1

        total_failures = sum(domain_failures.values())
        if total_failures > 0:
            boost_factor = 1.2
            for domain, count in domain_failures.items():
                if domain in mixture.domain_weights:
                    boost = 1.0 + (count / total_failures) * (boost_factor - 1.0)
                    mixture.domain_weights[domain] *= boost

        # Boost difficulty for hard examples
        difficulty_boost = {
            DifficultyLevel.D3.value: 1.2,
            DifficultyLevel.D4.value: 1.5,
            DifficultyLevel.D5.value: 2.0,
        }
        for example in hard_examples:
            diff = example.difficulty
            if diff in mixture.difficulty_weights and diff in difficulty_boost:
                mixture.difficulty_weights[diff] *= difficulty_boost[diff]

        mixture.normalize()
        return mixture

    def generate_hard_example_dataset(
        self,
        hard_examples: List[HardExample],
        output_path: str,
    ) -> None:
        """Generate a dataset from hard examples for training."""
        import json

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "examples": [e.to_dict() for e in hard_examples],
            "count": len(hard_examples),
            "generated_at": time.time(),
        }

        with open(output, "w") as f:
            json.dump(data, f, indent=2)
