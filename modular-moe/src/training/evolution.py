"""Population evolution and generation tracking for Prothynesis."""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum


class ModelStatus(Enum):
    """Model lifecycle status."""
    ACTIVE = "active"
    RETIRED = "retired"
    RETRAINING = "retraining"
    ARCHIVED = "archived"


@dataclass
class ModelFitness:
    """Multi-dimensional fitness profile for a model."""
    model_id: str
    individual_reasoning: float = 0.0
    knowledge: float = 0.0
    math: float = 0.0
    coding: float = 0.0
    verification: float = 0.0
    collaboration: float = 0.0
    memory: float = 0.0
    tool_use: float = 0.0
    reliability: float = 0.0
    composite_score: float = 0.0
    generation: int = 0
    last_updated: float = field(default_factory=time.time)

    def compute_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        if weights is None:
            weights = {
                "individual_reasoning": 0.2,
                "knowledge": 0.15,
                "math": 0.15,
                "coding": 0.1,
                "verification": 0.15,
                "collaboration": 0.1,
                "memory": 0.05,
                "tool_use": 0.05,
                "reliability": 0.05,
            }

        score = (
            weights.get("individual_reasoning", 0.0) * self.individual_reasoning +
            weights.get("knowledge", 0.0) * self.knowledge +
            weights.get("math", 0.0) * self.math +
            weights.get("coding", 0.0) * self.coding +
            weights.get("verification", 0.0) * self.verification +
            weights.get("collaboration", 0.0) * self.collaboration +
            weights.get("memory", 0.0) * self.memory +
            weights.get("tool_use", 0.0) * self.tool_use +
            weights.get("reliability", 0.0) * self.reliability
        )

        self.composite_score = score
        return score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenerationManifest:
    """Manifest for a training generation."""
    generation_id: int
    population_version: str
    dataset_version: str
    shared_core_hash: str
    unique_assignment_hash: str
    model_profiles: List[Dict[str, Any]]
    training_runs: List[Dict[str, Any]]
    benchmark_results: List[Dict[str, Any]]
    evolution_decisions: List[Dict[str, Any]]
    retained_models: List[str]
    retrained_models: List[str]
    replaced_profiles: List[str]
    timestamp: float = field(default_factory=time.time)
    code_version: str = ""
    environment_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, output_path: str) -> None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, input_path: str) -> "GenerationManifest":
        with open(input_path, "r") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class EvolutionDecision:
    """Decision made during population evolution."""
    decision_type: str  # "retain", "retrain", "replace", "specialize"
    model_id: str
    reason: str
    old_profile: Dict[str, Any]
    new_profile: Optional[Dict[str, Any]] = None
    fitness_before: Optional[float] = None
    fitness_after: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


class PopulationEvolution:
    """
    Manages population evolution across training generations.
    
    Supports:
    - Multi-dimensional fitness tracking
    - Model retention policies
    - Redundancy detection
    - Population reassignment
    - Curriculum evolution
    - Generation manifests
    """

    def __init__(self, config: dict):
        self.config = config
        self.generation = 0
        self.model_fitness: Dict[str, ModelFitness] = {}
        self.evolution_history: List[GenerationManifest] = []
        self.redundancy_threshold = config.get("population_evolution", {}).get("redundancy_threshold", 0.95)
        self.diversity_threshold = config.get("population_evolution", {}).get("diversity_threshold", 0.8)

    def register_generation(self, manifest: GenerationManifest) -> None:
        """Register a new generation."""
        self.generation = manifest.generation_id
        self.evolution_history.append(manifest)

    def update_fitness(self, model_id: str, metrics: Dict[str, float]) -> ModelFitness:
        """Update model fitness from evaluation metrics."""
        if model_id not in self.model_fitness:
            self.model_fitness[model_id] = ModelFitness(model_id=model_id, generation=self.generation)

        fitness = self.model_fitness[model_id]
        for key, value in metrics.items():
            if hasattr(fitness, key):
                setattr(fitness, key, value)

        fitness.compute_composite()
        fitness.last_updated = time.time()
        return fitness

    def get_model_fitness(self, model_id: str) -> Optional[ModelFitness]:
        """Get model fitness."""
        return self.model_fitness.get(model_id)

    def detect_redundancy(self, model_ids: List[str]) -> List[str]:
        """
        Detect redundant models.
        
        Returns list of model IDs that are potentially redundant.
        """
        if len(model_ids) < 2:
            return []

        redundant = []
        fitness_scores = [
            (mid, self.model_fitness.get(mid, ModelFitness(model_id=mid)).composite_score)
            for mid in model_ids
        ]

        # Group by similar fitness
        for i, (mid1, score1) in enumerate(fitness_scores):
            for mid2, score2 in fitness_scores[i + 1:]:
                if score1 > 0 and score2 > 0:
                    similarity = min(score1, score2) / max(score1, score2)
                    if similarity > self.redundancy_threshold:
                        redundant.append(mid2)

        return list(set(redundant))

    def select_survivors(
        self,
        model_ids: List[str],
        retention_ratio: float = 0.8,
    ) -> tuple[List[str], List[str]]:
        """
        Select models to retain vs. retire.
        
        Returns:
            (retained, retired)
        """
        scored = [
            (mid, self.model_fitness.get(mid, ModelFitness(model_id=mid)).composite_score)
            for mid in model_ids
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        keep_count = max(1, int(len(model_ids) * retention_ratio))
        retained = [mid for mid, _ in scored[:keep_count]]
        retired = [mid for mid, _ in scored[keep_count:]]

        return retained, retired

    def evolve_population(
        self,
        model_ids: List[str],
        hard_examples: List[Any],
    ) -> List[EvolutionDecision]:
        """
        Evolve the population based on fitness and hard examples.
        
        Returns list of evolution decisions.
        """
        decisions = []

        # Detect redundancy
        redundant = self.detect_redundancy(model_ids)

        # Select survivors
        retained, retired = self.select_survivors(model_ids)

        for model_id in model_ids:
            fitness = self.model_fitness.get(model_id, ModelFitness(model_id=model_id))

            if model_id in redundant:
                decisions.append(EvolutionDecision(
                    decision_type="replace",
                    model_id=model_id,
                    reason="Redundant with another model",
                    old_profile={"model_id": model_id},
                ))
            elif model_id in retained:
                if fitness.composite_score < 0.3:
                    decisions.append(EvolutionDecision(
                        decision_type="retrain",
                        model_id=model_id,
                        reason="Low fitness, retrain with updated curriculum",
                        old_profile={"model_id": model_id},
                    ))
                else:
                    decisions.append(EvolutionDecision(
                        decision_type="retain",
                        model_id=model_id,
                        reason="Sufficient fitness",
                        old_profile={"model_id": model_id},
                    ))
            else:
                decisions.append(EvolutionDecision(
                    decision_type="retire",
                    model_id=model_id,
                    reason="Below retention threshold",
                    old_profile={"model_id": model_id},
                ))

        return decisions

    def create_generation_manifest(
        self,
        generation_id: int,
        model_profiles: List[Dict[str, Any]],
        training_runs: List[Dict[str, Any]],
        benchmark_results: List[Dict[str, Any]],
        evolution_decisions: List[EvolutionDecision],
        retained_models: List[str],
        retrained_models: List[str],
        replaced_profiles: List[str],
    ) -> GenerationManifest:
        """Create a generation manifest."""
        # Compute hashes
        shared_core_hash = hashlib.sha256(
            json.dumps(model_profiles, sort_keys=True).encode()
        ).hexdigest()[:16]

        unique_hash = hashlib.sha256(
            json.dumps(training_runs, sort_keys=True).encode()
        ).hexdigest()[:16]

        manifest = GenerationManifest(
            generation_id=generation_id,
            population_version=f"gen_{generation_id:04d}",
            dataset_version=f"dataset_{generation_id:04d}",
            shared_core_hash=shared_core_hash,
            unique_assignment_hash=unique_hash,
            model_profiles=model_profiles,
            training_runs=training_runs,
            benchmark_results=benchmark_results,
            evolution_decisions=[d.to_dict() for d in evolution_decisions],
            retained_models=retained_models,
            retrained_models=retrained_models,
            replaced_profiles=replaced_profiles,
        )

        self.register_generation(manifest)
        return manifest

    def get_population_diversity_report(self) -> Dict[str, Any]:
        """Get population diversity report."""
        if not self.model_fitness:
            return {"status": "no_data"}

        fitnesses = [f.composite_score for f in self.model_fitness.values()]
        avg_fitness = sum(fitnesses) / len(fitnesses)
        fitness_std = (sum((f - avg_fitness) ** 2 for f in fitnesses) / len(fitnesses)) ** 0.5

        return {
            "generation": self.generation,
            "model_count": len(self.model_fitness),
            "average_fitness": avg_fitness,
            "fitness_std": fitness_std,
            "diversity_score": min(1.0, fitness_std / max(0.01, avg_fitness)),
            "status": "healthy" if fitness_std > 0.05 else "low_diversity",
        }


class GenerationTracker:
    """Tracks training generations and their manifests."""

    def __init__(self, manifest_dir: str = "manifests"):
        self.manifest_dir = Path(manifest_dir)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.manifests: Dict[int, GenerationManifest] = {}
        self.current_generation = 0

    def start_generation(self, config: Dict[str, Any]) -> GenerationManifest:
        """Start a new generation."""
        self.current_generation += 1
        manifest = GenerationManifest(
            generation_id=self.current_generation,
            population_version=f"gen_{self.current_generation:04d}",
            dataset_version="",
            shared_core_hash="",
            unique_assignment_hash="",
            model_profiles=[],
            training_runs=[],
            benchmark_results=[],
            evolution_decisions=[],
            retained_models=[],
            retrained_models=[],
            replaced_profiles=[],
            environment_metadata={
                "config": config,
                "started_at": time.time(),
            },
        )
        self.manifests[self.current_generation] = manifest
        return manifest

    def complete_generation(self, manifest: GenerationManifest) -> None:
        """Complete a generation and save manifest."""
        manifest.timestamp = time.time()
        output_path = self.manifest_dir / f"generation_{manifest.generation_id:04d}.json"
        manifest.save(output_path)

    def load_generation(self, generation_id: int) -> Optional[GenerationManifest]:
        """Load a generation manifest."""
        if generation_id in self.manifests:
            return self.manifests[generation_id]

        input_path = self.manifest_dir / f"generation_{generation_id:04d}.json"
        if input_path.exists():
            manifest = GenerationManifest.load(input_path)
            self.manifests[generation_id] = manifest
            return manifest

        return None

    def get_latest_generation(self) -> Optional[GenerationManifest]:
        """Get the latest generation manifest."""
        if not self.manifests:
            return None
        return self.manifests[max(self.manifests.keys())]
