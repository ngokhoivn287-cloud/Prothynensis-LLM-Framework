"""Dataset scheduler for Prothynesis."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


@dataclass
class DatasetProfile:
    """Profile for a dataset."""
    name: str
    domain: str
    difficulty: str  # D0-D5
    task_type: str
    language: str
    quality_score: float = 1.0
    size: int = 0
    source: str = ""
    license: str = ""
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "domain": self.domain,
            "difficulty": self.difficulty,
            "task_type": self.task_type,
            "language": self.language,
            "quality_score": self.quality_score,
            "size": self.size,
            "source": self.source,
            "license": self.license,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "DatasetProfile":
        return cls(**d)


@dataclass
class ModelMixture:
    """Training mixture for a model."""
    model_id: str
    domain_weights: Dict[str, float] = field(default_factory=dict)
    difficulty_weights: Dict[str, float] = field(default_factory=dict)
    task_weights: Dict[str, float] = field(default_factory=dict)
    language_weights: Dict[str, float] = field(default_factory=dict)
    
    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        total = sum(self.domain_weights.values())
        if total > 0:
            self.domain_weights = {k: v / total for k, v in self.domain_weights.items()}
        
        total = sum(self.difficulty_weights.values())
        if total > 0:
            self.difficulty_weights = {k: v / total for k, v in self.difficulty_weights.items()}
        
        total = sum(self.task_weights.values())
        if total > 0:
            self.task_weights = {k: v / total for k, v in self.task_weights.items()}
        
        total = sum(self.language_weights.values())
        if total > 0:
            self.language_weights = {k: v / total for k, v in self.language_weights.items()}
    
    def validate_coverage(self, config: dict) -> List[str]:
        """Validate minimum coverage constraints."""
        issues = []
        
        if config.get("min_coverage_all_domains", False):
            # Ensure all core domains have non-zero weight
            core_domains = {"mathematics", "programming", "science", "reasoning", "knowledge"}
            missing = core_domains - set(self.domain_weights.keys())
            if missing:
                issues.append(f"Missing core domains: {missing}")
        
        min_reasoning = config.get("minimum_reasoning_exposure", 0.0)
        reasoning_weight = self.task_weights.get("reasoning", 0.0)
        if reasoning_weight < min_reasoning:
            issues.append(f"Reasoning exposure {reasoning_weight:.2f} < minimum {min_reasoning:.2f}")
        
        min_verification = config.get("minimum_verification_exposure", 0.0)
        verification_weight = self.task_weights.get("verification", 0.0)
        if verification_weight < min_verification:
            issues.append(f"Verification exposure {verification_weight:.2f} < minimum {min_verification:.2f}")
        
        return issues
    
    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "domain_weights": self.domain_weights,
            "difficulty_weights": self.difficulty_weights,
            "task_weights": self.task_weights,
            "language_weights": self.language_weights,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ModelMixture":
        return cls(**d)


class DatasetScheduler:
    """
    Dataset scheduler for Prothynesis.
    
    Generates training manifests by combining:
    - domain weights
    - difficulty weights
    - task weights
    - language weights
    - model specialization profile
    - global coverage constraints
    """
    
    CORE_DOMAINS = [
        "mathematics", "programming", "science", "reasoning", "knowledge",
        "systems", "languages", "humanities", "verification", "research"
    ]
    
    CORE_TASKS = [
        "reasoning", "verification", "coding", "knowledge", "generation",
        "critique", "problem_solving", "explanation", "synthesis"
    ]
    
    DIFFICULTIES = ["D0", "D1", "D2", "D3", "D4", "D5"]
    
    def __init__(self, config: dict):
        self.config = config
        self.dataset_profiles: List[DatasetProfile] = []
        self.model_mixtures: Dict[str, ModelMixture] = {}
    
    def register_dataset(self, profile: DatasetProfile) -> None:
        """Register a dataset profile."""
        self.dataset_profiles.append(profile)
    
    def generate_model_mixture(self, model_id: str, seed: Optional[int] = None) -> ModelMixture:
        """Generate a model-specific mixture with configurable diversity."""
        if seed is not None:
            random.seed(seed)
        
        mixture = ModelMixture(model_id=model_id)
        
        # Base domain weights with perturbation
        base_domain = {d: 0.10 for d in self.CORE_DOMAINS}
        # Add specialization emphasis
        specialization = random.choice(self.CORE_DOMAINS)
        base_domain[specialization] = 0.25
        # Perturb
        for d in base_domain:
            base_domain[d] *= random.uniform(0.8, 1.2)
        mixture.domain_weights = base_domain
        
        # Difficulty weights
        difficulty = {
            "D0": random.uniform(0.05, 0.15),
            "D1": random.uniform(0.10, 0.20),
            "D2": random.uniform(0.25, 0.35),
            "D3": random.uniform(0.20, 0.30),
            "D4": random.uniform(0.10, 0.20),
            "D5": random.uniform(0.03, 0.10),
        }
        mixture.difficulty_weights = difficulty
        
        # Task weights
        task = {t: random.uniform(0.05, 0.20) for t in self.CORE_TASKS}
        task["reasoning"] = random.uniform(0.15, 0.30)
        mixture.task_weights = task
        
        # Language weights (multilingual)
        languages = ["en", "vi", "zh", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar", "hi"]
        lang = {lang: random.uniform(0.02, 0.15) for lang in languages}
        mixture.language_weights = lang
        
        mixture.normalize()
        
        # Validate coverage
        issues = mixture.validate_coverage(self.config.get("datasets", {}))
        if issues:
            # Fix issues by adding minimum weights
            if "Missing core domains" in str(issues):
                for d in self.CORE_DOMAINS:
                    if d not in mixture.domain_weights:
                        mixture.domain_weights[d] = 0.01
        
        return mixture
    
    def generate_pool_mixtures(self, pool_size: int, seed: int = 42) -> Dict[str, ModelMixture]:
        """Generate mixtures for entire pool."""
        mixtures = {}
        for i in range(pool_size):
            model_id = f"solver_{i:05d}"
            mixtures[model_id] = self.generate_model_mixture(model_id, seed=seed + i)
        return mixtures
    
    def generate_training_manifest(self, model_id: str, mixture: ModelMixture) -> dict:
        """Generate a deterministic training manifest for a model."""
        manifest = {
            "model_id": model_id,
            "mixture": mixture.to_dict(),
            "datasets": [],
            "sampling_params": {
                "temperature": self.config.get("datasets", {}).get("sampling_temperature", 1.0),
                "repetition_penalty": 1.1,
            },
            "coverage_constraints": self.config.get("datasets", {}),
        }
        
        # Match datasets to mixture
        for profile in self.dataset_profiles:
            domain_weight = mixture.domain_weights.get(profile.domain, 0.0)
            difficulty_weight = mixture.difficulty_weights.get(profile.difficulty, 0.0)
            task_weight = mixture.task_weights.get(profile.task_type, 0.0)
            
            combined_weight = domain_weight * difficulty_weight * task_weight * profile.quality_score
            
            if combined_weight > 0.001:
                manifest["datasets"].append({
                    "name": profile.name,
                    "weight": round(combined_weight, 4),
                    "domain": profile.domain,
                    "difficulty": profile.difficulty,
                    "task_type": profile.task_type,
                    "language": profile.language,
                })
        
        # Normalize dataset weights
        total_weight = sum(d["weight"] for d in manifest["datasets"])
        if total_weight > 0:
            for d in manifest["datasets"]:
                d["weight"] = round(d["weight"] / total_weight, 4)
        
        return manifest


def create_scheduler(config: dict) -> DatasetScheduler:
    """Factory function to create a dataset scheduler."""
    return DatasetScheduler(config)
