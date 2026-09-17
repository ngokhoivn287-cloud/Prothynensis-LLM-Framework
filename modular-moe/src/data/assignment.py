"""Dataset assignment and ownership engine for Prothynesis."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from .scheduler import DatasetProfile, DatasetScheduler, ModelMixture


@dataclass
class SampleRecord:
    """Record for a single dataset sample."""
    sample_id: str
    text: str
    token_count: int
    domain: str
    difficulty: str
    task_type: str
    language: str
    quality_score: float
    source: str
    is_shared: bool = False
    owner_model_id: Optional[str] = None
    is_replicated: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class OwnershipReport:
    """Report on dataset ownership and duplication."""
    total_samples: int = 0
    shared_core_count: int = 0
    unique_pool_count: int = 0
    assigned_count: int = 0
    unassigned_count: int = 0
    duplicate_count: int = 0
    duplicate_percentage: float = 0.0
    replicated_count: int = 0
    per_model_counts: Dict[str, int] = field(default_factory=dict)
    domain_coverage: Dict[str, int] = field(default_factory=dict)
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)
    task_distribution: Dict[str, int] = field(default_factory=dict)
    language_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class DatasetAssignmentEngine:
    """
    Assigns dataset samples to models with:
    - Shared core (~1%) replicated to all models
    - Unique experience pool (~99%) with stratified allocation
    - Ownership tracking to prevent duplicates
    """

    def __init__(self, config: dict):
        self.config = config
        self.shared_core_ratio = config.get("shared_core", {}).get("ratio", 0.01)
        self.unique_ratio = config.get("unique_experience", {}).get("ratio", 0.99)
        self.critical_replication = config.get("critical_replication", {})
        self.samples: List[SampleRecord] = []
        self.ownership: Dict[str, str] = {}  # sample_id -> model_id
        self.shared_sample_ids: set = set()
        self.replicated_sample_ids: set = set()

    def load_samples(self, samples: List[SampleRecord]) -> None:
        """Load samples into the engine."""
        self.samples = list(samples)

    def build_shared_core(self) -> List[SampleRecord]:
        """Select high-quality samples for the shared core."""
        if not self.samples:
            return []

        # Sort by quality score descending, then by diversity
        sorted_samples = sorted(
            self.samples,
            key=lambda s: (
                s.quality_score,
                s.domain != "knowledge",
                s.task_type != "reasoning",
            ),
            reverse=True,
        )

        shared_count = max(1, int(len(self.samples) * self.shared_core_ratio))
        shared_core = sorted_samples[:shared_count]

        # Mark as shared
        for sample in shared_core:
            sample.is_shared = True
            self.shared_sample_ids.add(sample.sample_id)

        return shared_core

    def assign_unique_experience(
        self,
        model_ids: List[str],
        scheduler: DatasetScheduler,
        seed: int = 42,
    ) -> Dict[str, List[SampleRecord]]:
        """
        Assign unique samples to models with stratified allocation.
        
        Each model gets a unique subset of the remaining samples,
        with broad domain/difficulty/task/language coverage.
        """
        if not self.samples:
            return {model_id: [] for model_id in model_ids}

        random.seed(seed)

        # Get non-shared, non-replicated samples
        available = [s for s in self.samples if not s.is_shared and not s.is_replicated]

        # Generate mixtures for each model
        model_mixtures = {}
        for i, model_id in enumerate(model_ids):
            mixture = scheduler.generate_model_mixture(model_id, seed=seed + i)
            model_mixtures[model_id] = mixture

        # Stratified allocation
        assignments = {model_id: [] for model_id in model_ids}

        # Group available samples by domain
        domain_groups: Dict[str, List[SampleRecord]] = {}
        for sample in available:
            domain_groups.setdefault(sample.domain, []).append(sample)

        # If no domain-specific samples, treat all as general
        if not domain_groups:
            domain_groups["general"] = available

        # For each model, allocate samples proportionally
        for model_id in model_ids:
            mixture = model_mixtures[model_id]
            target_count = len(available) // len(model_ids)

            # Calculate allocation per domain based on mixture weights
            domain_allocation = {}
            remaining = target_count
            domains = list(mixture.domain_weights.keys())
            for i, domain in enumerate(domains):
                if i == len(domains) - 1:
                    domain_allocation[domain] = remaining
                else:
                    alloc = int(target_count * mixture.domain_weights.get(domain, 0.0))
                    domain_allocation[domain] = alloc
                    remaining -= alloc

            # Assign samples from each domain
            for domain, count in domain_allocation.items():
                domain_samples = domain_groups.get(domain, [])
                if not domain_samples:
                    # Fallback: assign from any available domain
                    domain_samples = available

                # Shuffle within domain for diversity
                random.shuffle(domain_samples)

                # Take up to count samples that haven't been assigned
                assigned = 0
                for sample in domain_samples:
                    if assigned >= count:
                        break
                    if sample.sample_id not in self.ownership:
                        self.ownership[sample.sample_id] = model_id
                        sample.owner_model_id = model_id
                        assignments[model_id].append(sample)
                        assigned += 1

        return assignments

    def assign_critical_replication(self, model_ids: List[str], samples: List[SampleRecord]) -> None:
        """Assign critical replicated samples to all models."""
        if not self.critical_replication.get("enabled", False):
            return

        ratio = self.critical_replication.get("ratio", 0.001)
        critical_count = max(1, int(len(self.samples) * ratio))

        # Select critical samples (highest quality, diverse)
        available = [s for s in self.samples if not s.is_shared and not s.is_replicated]
        available.sort(key=lambda s: s.quality_score, reverse=True)
        critical = available[:critical_count]

        for sample in critical:
            sample.is_replicated = True
            self.replicated_sample_ids.add(sample.sample_id)
            # Assign to all models
            for model_id in model_ids:
                self.ownership[sample.sample_id] = model_id

    def check_ownership(self) -> OwnershipReport:
        """Check dataset ownership and duplication."""
        report = OwnershipReport()
        report.total_samples = len(self.samples)
        report.shared_core_count = len(self.shared_sample_ids)
        report.replicated_count = len(self.replicated_sample_ids)

        unique_samples = [s for s in self.samples if not s.is_shared and not s.is_replicated]
        report.unique_pool_count = len(unique_samples)

        # Count assignments
        model_counts: Dict[str, int] = {}
        domain_coverage: Dict[str, int] = {}
        difficulty_dist: Dict[str, int] = {}
        task_dist: Dict[str, int] = {}
        lang_dist: Dict[str, int] = {}

        for sample in self.samples:
            if sample.is_shared:
                continue
            if sample.is_replicated:
                continue

            owner = sample.owner_model_id
            if owner:
                report.assigned_count += 1
                model_counts[owner] = model_counts.get(owner, 0) + 1
                domain_coverage[sample.domain] = domain_coverage.get(sample.domain, 0) + 1
                difficulty_dist[sample.difficulty] = difficulty_dist.get(sample.difficulty, 0) + 1
                task_dist[sample.task_type] = task_dist.get(sample.task_type, 0) + 1
                lang_dist[sample.language] = lang_dist.get(sample.language, 0) + 1
            else:
                report.unassigned_count += 1

        # Check for duplicates in unique pool
        sample_owners: Dict[str, List[str]] = {}
        for sample_id, model_id in self.ownership.items():
            if sample_id not in self.shared_sample_ids and sample_id not in self.replicated_sample_ids:
                sample_owners.setdefault(sample_id, []).append(model_id)

        duplicate_count = sum(1 for owners in sample_owners.values() if len(owners) > 1)
        report.duplicate_count = duplicate_count
        if report.unique_pool_count > 0:
            report.duplicate_percentage = (duplicate_count / report.unique_pool_count) * 100.0

        report.per_model_counts = model_counts
        report.domain_coverage = domain_coverage
        report.difficulty_distribution = difficulty_dist
        report.task_distribution = task_dist
        report.language_distribution = lang_dist

        return report

    def check_population_diversity(self, model_ids: List[str]) -> Dict[str, Any]:
        """Check diversity across model datasets."""
        if not model_ids:
            return {"warning": "No models"}

        # Get samples per model
        model_samples = {model_id: [] for model_id in model_ids}
        for sample in self.samples:
            if sample.owner_model_id in model_samples:
                model_samples[sample.owner_model_id].append(sample)

        # Calculate domain similarity between models
        domain_vectors = {}
        for model_id in model_ids:
            samples = model_samples[model_id]
            domain_counts = {}
            for s in samples:
                domain_counts[s.domain] = domain_counts.get(s.domain, 0) + 1
            total = len(samples) if samples else 1
            domain_vectors[model_id] = {d: c / total for d, c in domain_counts.items()}

        # Pairwise cosine similarity
        similarities = []
        model_list = list(model_ids)
        for i in range(len(model_list)):
            for j in range(i + 1, len(model_list)):
                v1 = domain_vectors[model_list[i]]
                v2 = domain_vectors[model_list[j]]
                all_domains = set(v1.keys()) | set(v2.keys())
                dot = sum(v1.get(d, 0) * v2.get(d, 0) for d in all_domains)
                norm1 = sum(v1.get(d, 0) ** 2 for d in all_domains) ** 0.5
                norm2 = sum(v2.get(d, 0) ** 2 for d in all_domains) ** 0.5
                if norm1 > 0 and norm2 > 0:
                    sim = dot / (norm1 * norm2)
                    similarities.append(sim)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        result = {
            "model_count": len(model_ids),
            "average_domain_similarity": avg_similarity,
            "population_diversity_warning": avg_similarity > 0.9,
            "pairwise_similarities": similarities[:10],  # Sample
        }

        return result

    def save_assignment(self, output_path: str) -> None:
        """Save assignment metadata."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "shared_sample_ids": list(self.shared_sample_ids),
            "replicated_sample_ids": list(self.replicated_sample_ids),
            "ownership": self.ownership,
            "samples": [s.to_dict() for s in self.samples],
        }

        with open(output, "w") as f:
            json.dump(data, f, indent=2)

    def load_assignment(self, input_path: str) -> None:
        """Load assignment metadata."""
        with open(input_path, "r") as f:
            data = json.load(f)

        self.shared_sample_ids = set(data.get("shared_sample_ids", []))
        self.replicated_sample_ids = set(data.get("replicated_sample_ids", []))
        self.ownership = data.get("ownership", {})
        self.samples = [SampleRecord(**s) for s in data.get("samples", [])]


def create_assignment_engine(config: dict) -> DatasetAssignmentEngine:
    """Factory function to create assignment engine."""
    return DatasetAssignmentEngine(config)
