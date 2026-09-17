"""Uncertainty and calibration for Prothynesis models.

Tracks:
- confidence vs actual correctness
- calibration error
- uncertainty levels
- evidence quality
- error diagnosis
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class UncertaintyLevel(Enum):
    CERTAIN = "certain"
    LIKELY = "likely"
    UNCERTAIN = "uncertain"
    UNKNOWN = "unknown"
    NOT_VERIFIED = "not_verified"


@dataclass
class CalibrationRecord:
    """Single confidence-accuracy observation."""
    prediction_id: str
    model_id: str
    confidence: float
    actual_correct: bool
    task_domain: str
    task_difficulty: float
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UncertaintyProfile:
    """Uncertainty profile for a model on a specific task."""
    model_id: str
    task_id: str
    uncertainty_level: str = UncertaintyLevel.NOT_VERIFIED.value
    confidence: float = 0.0
    evidence_quality: float = 0.0
    assumptions: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    verification_plan: List[str] = field(default_factory=list)
    verification_status: str = "pending"
    error_diagnosis: Optional[str] = None
    self_correction_attempted: bool = False
    calibration_error: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class CalibrationTracker:
    """Tracks confidence calibration across predictions."""

    def __init__(self, config: dict):
        self.config = config
        self.bin_count = config.get("calibration", {}).get("bin_count", 10)
        self._records: List[CalibrationRecord] = []

    def record_prediction(
        self,
        prediction_id: str,
        model_id: str,
        confidence: float,
        actual_correct: bool,
        task_domain: str = "general",
        task_difficulty: float = 0.5,
    ) -> CalibrationRecord:
        record = CalibrationRecord(
            prediction_id=prediction_id,
            model_id=model_id,
            confidence=confidence,
            actual_correct=actual_correct,
            task_domain=task_domain,
            task_difficulty=task_difficulty,
        )
        self._records.append(record)
        return record

    def compute_calibration_error(self, model_id: Optional[str] = None) -> float:
        records = self._records
        if model_id:
            records = [r for r in records if r.model_id == model_id]
        if not records:
            return 0.0
        bins: Dict[int, List[CalibrationRecord]] = {i: [] for i in range(self.bin_count)}
        for r in records:
            bin_idx = min(int(r.confidence * self.bin_count), self.bin_count - 1)
            bins[bin_idx].append(r)
        total_error = 0.0
        total_count = 0
        for bin_idx, bin_records in bins.items():
            if not bin_records:
                continue
            bin_center = (bin_idx + 0.5) / self.bin_count
            bin_accuracy = sum(1 for r in bin_records if r.actual_correct) / len(bin_records)
            total_error += abs(bin_center - bin_accuracy) * len(bin_records)
            total_count += len(bin_records)
        if total_count == 0:
            return 0.0
        return total_error / total_count

    def get_reliability_diagram(self, model_id: Optional[str] = None) -> Dict[str, List[float]]:
        records = self._records
        if model_id:
            records = [r for r in records if r.model_id == model_id]
        bins: Dict[int, Dict[str, float]] = {i: {"count": 0, "correct": 0} for i in range(self.bin_count)}
        for r in records:
            bin_idx = min(int(r.confidence * self.bin_count), self.bin_count - 1)
            bins[bin_idx]["count"] += 1
            if r.actual_correct:
                bins[bin_idx]["correct"] += 1
        confidences = []
        accuracies = []
        counts = []
        for i in range(self.bin_count):
            if bins[i]["count"] == 0:
                continue
            confidences.append((i + 0.5) / self.bin_count)
            accuracies.append(bins[i]["correct"] / bins[i]["count"])
            counts.append(bins[i]["count"])
        return {"confidences": confidences, "accuracies": accuracies, "counts": counts}

    def get_model_calibration_summary(self, model_id: str) -> Dict[str, Any]:
        records = [r for r in self._records if r.model_id == model_id]
        if not records:
            return {"model_id": model_id, "calibration_error": 0.0, "total_predictions": 0}
        return {
            "model_id": model_id,
            "calibration_error": self.compute_calibration_error(model_id),
            "total_predictions": len(records),
            "average_confidence": sum(r.confidence for r in records) / len(records),
            "accuracy": sum(1 for r in records if r.actual_correct) / len(records),
        }

    def get_overcalibrated_models(self, threshold: float = 0.1) -> List[str]:
        overcalibrated = []
        seen = set()
        for r in self._records:
            if r.model_id in seen:
                continue
            seen.add(r.model_id)
            error = self.compute_calibration_error(r.model_id)
            if error > threshold:
                overcalibrated.append(r.model_id)
        return overcalibrated


class UncertaintyEstimator:
    """Estimates uncertainty levels for model outputs."""

    def __init__(self, config: dict):
        self.config = config
        self.calibration_tracker = CalibrationTracker(config)

    def estimate_uncertainty(
        self,
        confidence: float,
        evidence_quality: float,
        verification_status: str,
        contradictions: List[str],
    ) -> UncertaintyProfile:
        """Estimate uncertainty from multiple signals."""
        profile = UncertaintyProfile(
            model_id="",
            task_id="",
            confidence=confidence,
            evidence_quality=evidence_quality,
            contradictions=contradictions,
        )

        if contradictions:
            profile.uncertainty_level = UncertaintyLevel.UNCERTAIN.value
            profile.verification_status = "failed_contradiction"
            return profile

        if verification_status == "pass" and evidence_quality >= 0.8:
            profile.uncertainty_level = UncertaintyLevel.CERTAIN.value
            profile.verification_status = "verified"
            return profile

        if verification_status == "pass":
            profile.uncertainty_level = UncertaintyLevel.LIKELY.value
            profile.verification_status = "verified"
            return profile

        if verification_status == "pending":
            profile.uncertainty_level = UncertaintyLevel.UNKNOWN.value
            profile.verification_status = "pending"
            return profile

        if verification_status == "fail":
            profile.uncertainty_level = UncertaintyLevel.NOT_VERIFIED.value
            profile.verification_status = "failed"
            return profile

        if confidence >= 0.8 and evidence_quality >= 0.6:
            profile.uncertainty_level = UncertaintyLevel.LIKELY.value
            profile.verification_status = "unverified"
            return profile

        profile.uncertainty_level = UncertaintyLevel.UNCERTAIN.value
        profile.verification_status = "uncertain"
        return profile

    def get_calibrated_confidence(self, raw_confidence: float, model_id: str) -> float:
        summary = self.calibration_tracker.get_model_calibration_summary(model_id)
        if summary["total_predictions"] < 5:
            return raw_confidence
        calibration_error = summary["calibration_error"]
        if calibration_error > 0.1:
            return raw_confidence * (1.0 - calibration_error)
        return raw_confidence
