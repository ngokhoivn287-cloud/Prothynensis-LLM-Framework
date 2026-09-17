"""Reputation and contribution tracking for Prothynesis population models.

Tracks per-model reliability across:
- global performance
- capability domains
- task families
- difficulty levels
- verification accuracy
- hallucination/error rate
- tool-use success
- collaboration usefulness
- contribution ledger

Reputation feeds into:
- recruitment
- candidate scoring
- evidence-weighted consensus
- knowledge reuse prioritization
"""

from __future__ import annotations

import json
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum


class ReputationTier(Enum):
    """Model reliability tier."""
    UNINITIALIZED = "uninitialized"
    NOVEL = "novel"
    LEARNING = "learning"
    COMPETENT = "competent"
    RELIABLE = "reliable"
    EXPERT = "expert"
    TRUSTED = "trusted"


@dataclass
class DomainReputation:
    """Reputation within a specific capability domain."""
    domain: str
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    total_confidence: float = 0.0
    total_accuracy: float = 0.0
    calibration_error: float = 0.0
    last_updated: float = field(default_factory=time.time)

    @property
    def accuracy(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.successes / self.attempts

    @property
    def average_confidence(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_confidence / self.attempts

    def record_attempt(self, success: bool, confidence: float, actual_correct: bool) -> None:
        self.attempts += 1
        self.last_updated = time.time()
        self.total_confidence += confidence
        if success:
            self.successes += 1
        else:
            self.failures += 1
        self.total_accuracy += (1.0 if actual_correct else 0.0)
        if self.attempts > 1:
            self.calibration_error = abs(self.average_confidence - (self.total_accuracy / self.attempts))


@dataclass
class ContributionLedger:
    """Records how useful a model was for specific tasks."""
    model_id: str
    tasks_participated: int = 0
    tasks_contributed: int = 0
    verifications_performed: int = 0
    verifications_correct: int = 0
    critiques_helpful: int = 0
    critiques_total: int = 0
    information_gain_events: int = 0
    hard_examples_found: int = 0
    tool_calls_successful: int = 0
    tool_calls_total: int = 0
    peer_endorsements: int = 0
    peer_rejections: int = 0
    revision_requests: int = 0
    last_contributed: float = field(default_factory=time.time)

    @property
    def contribution_rate(self) -> float:
        if self.tasks_participated == 0:
            return 0.0
        return self.tasks_contributed / self.tasks_participated

    @property
    def verification_accuracy(self) -> float:
        if self.verifications_performed == 0:
            return 0.0
        return self.verifications_correct / self.verifications_performed

    def record_participation(self, contributed: bool) -> None:
        self.tasks_participated += 1
        if contributed:
            self.tasks_contributed += 1
        self.last_contributed = time.time()

    def record_verification(self, correct: bool) -> None:
        self.verifications_performed += 1
        if correct:
            self.verifications_correct += 1

    def record_critique(self, helpful: bool) -> None:
        self.critiques_total += 1
        if helpful:
            self.critiques_helpful += 1

    def record_tool_call(self, successful: bool) -> None:
        self.tool_calls_total += 1
        if successful:
            self.tool_calls_successful += 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelReputation:
    """Persistent reputation profile for a model."""
    model_id: str
    model_family: str = "general_solver"
    tier: str = ReputationTier.UNINITIALIZED.value
    global_attempts: int = 0
    global_successes: int = 0
    global_failures: int = 0
    total_confidence: float = 0.0
    total_accuracy: float = 0.0
    calibration_error: float = 0.0
    hallucination_rate: float = 0.0
    error_rate: float = 0.0
    domain_reputations: Dict[str, DomainReputation] = field(default_factory=dict)
    task_family_reputations: Dict[str, DomainReputation] = field(default_factory=dict)
    difficulty_reputations: Dict[str, DomainReputation] = field(default_factory=dict)
    contribution_ledger: ContributionLedger = field(default_factory=lambda: ContributionLedger(model_id=""))
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.contribution_ledger.model_id:
            self.contribution_ledger.model_id = self.model_id

    @property
    def global_accuracy(self) -> float:
        if self.global_attempts == 0:
            return 0.0
        return self.global_successes / self.global_attempts

    @property
    def global_average_confidence(self) -> float:
        if self.global_attempts == 0:
            return 0.0
        return self.total_confidence / self.global_attempts

    @property
    def reliability_score(self) -> float:
        if self.global_attempts < 5:
            return 0.5
        accuracy = self.global_accuracy
        calibration = max(0.0, 1.0 - self.calibration_error)
        return accuracy * 0.7 + calibration * 0.3

    def record_global_attempt(
        self,
        success: bool,
        confidence: float,
        actual_correct: bool,
        is_hallucination: bool = False,
        is_error: bool = False,
    ) -> None:
        self.global_attempts += 1
        self.last_updated = time.time()
        self.total_confidence += confidence
        self.total_accuracy += (1.0 if actual_correct else 0.0)
        if success:
            self.global_successes += 1
        else:
            self.global_failures += 1
        if is_hallucination:
            self.hallucination_rate = (
                (self.hallucination_rate * (self.global_attempts - 1) + 1.0) / self.global_attempts
            )
        if is_error:
            self.error_rate = (
                (self.error_rate * (self.global_attempts - 1) + 1.0) / self.global_attempts
            )
        if self.global_attempts > 1:
            self.calibration_error = abs(self.global_average_confidence - (self.total_accuracy / self.global_attempts))

    def record_domain_attempt(
        self,
        domain: str,
        success: bool,
        confidence: float,
        actual_correct: bool,
    ) -> None:
        if domain not in self.domain_reputations:
            self.domain_reputations[domain] = DomainReputation(domain=domain)
        self.domain_reputations[domain].record_attempt(success, confidence, actual_correct)

    def get_domain_accuracy(self, domain: str) -> float:
        if domain not in self.domain_reputations:
            return 0.0
        return self.domain_reputations[domain].accuracy

    def update_tier(self) -> None:
        if self.global_attempts < 3:
            self.tier = ReputationTier.NOVEL.value
        elif self.global_accuracy < 0.3:
            self.tier = ReputationTier.LEARNING.value
        elif self.global_accuracy < 0.6:
            self.tier = ReputationTier.COMPETENT.value
        elif self.global_accuracy < 0.8:
            self.tier = ReputationTier.RELIABLE.value
        elif self.global_accuracy < 0.95:
            self.tier = ReputationTier.EXPERT.value
        else:
            self.tier = ReputationTier.TRUSTED.value

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain_reputations"] = {k: v.to_dict() for k, v in self.domain_reputations.items()}
        d["task_family_reputations"] = {k: v.to_dict() for k, v in self.task_family_reputations.items()}
        d["difficulty_reputations"] = {k: v.to_dict() for k, v in self.difficulty_reputations.items()}
        d["contribution_ledger"] = self.contribution_ledger.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelReputation":
        domain_reps = {
            k: DomainReputation(**v) for k, v in d.pop("domain_reputations", {}).items()
        }
        task_reps = {
            k: DomainReputation(**v) for k, v in d.pop("task_family_reputations", {}).items()
        }
        diff_reps = {
            k: DomainReputation(**v) for k, v in d.pop("difficulty_reputations", {}).items()
        }
        ledger_data = d.pop("contribution_ledger", {})
        obj = cls(**d)
        obj.domain_reputations = domain_reps
        obj.task_family_reputations = task_reps
        obj.difficulty_reputations = diff_reps
        if ledger_data:
            obj.contribution_ledger = ContributionLedger(**ledger_data)
        return obj


class ReputationStore:
    """Persistent store for model reputation profiles."""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self._reputations: Dict[str, ModelReputation] = {}

    def get_or_create(self, model_id: str, model_family: str = "general_solver") -> ModelReputation:
        if model_id not in self._reputations:
            self._reputations[model_id] = ModelReputation(
                model_id=model_id,
                model_family=model_family,
            )
        return self._reputations[model_id]

    def update_reputation(self, reputation: ModelReputation) -> None:
        reputation.update_tier()
        self._reputations[reputation.model_id] = reputation
        if self.storage_path:
            self._persist()

    def get_reputation(self, model_id: str) -> Optional[ModelReputation]:
        return self._reputations.get(model_id)

    def get_most_reliable(self, domain: Optional[str] = None, min_attempts: int = 5) -> List[ModelReputation]:
        candidates = []
        for rep in self._reputations.values():
            if rep.global_attempts < min_attempts:
                continue
            if domain:
                acc = rep.get_domain_accuracy(domain)
                if acc == 0.0 and domain not in rep.domain_reputations:
                    continue
                candidates.append((acc, rep))
            else:
                candidates.append((rep.reliability_score, rep))
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [rep for _, rep in candidates]

    def get_contribution_ranking(self) -> List[Tuple[str, float]]:
        rankings = []
        for model_id, rep in self._reputations.items():
            rankings.append((model_id, rep.contribution_ledger.contribution_rate))
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings

    def _persist(self) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {mid: rep.to_dict() for mid, rep in self._reputations.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        with open(self.storage_path, "r") as f:
            data = json.load(f)
        for mid, d in data.items():
            self._reputations[mid] = ModelReputation.from_dict(d)

    def to_dict(self) -> dict:
        return {mid: rep.to_dict() for mid, rep in self._reputations.items()}
