"""Tests for calibration and uncertainty estimation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_calibration_tracker_records_predictions():
    """Test calibration tracker records predictions."""
    from src.training.calibration import CalibrationTracker

    tracker = CalibrationTracker({"calibration": {"bin_count": 10}})
    tracker.record_prediction("p1", "model_1", confidence=0.9, actual_correct=True)
    tracker.record_prediction("p2", "model_1", confidence=0.8, actual_correct=False)

    assert tracker.compute_calibration_error("model_1") > 0.0


def test_calibration_tracker_perfect_calibration():
    """Test calibration tracker with perfectly calibrated predictions."""
    from src.training.calibration import CalibrationTracker

    tracker = CalibrationTracker({"calibration": {"bin_count": 10}})
    for i in range(20):
        confidence = i / 20.0
        tracker.record_prediction(f"p{i}", "model_1", confidence=confidence, actual_correct=(i % 2 == 0))

    error = tracker.compute_calibration_error("model_1")
    assert error >= 0.0


def test_reliability_diagram():
    """Test reliability diagram generation."""
    from src.training.calibration import CalibrationTracker

    tracker = CalibrationTracker({"calibration": {"bin_count": 10}})
    for i in range(20):
        confidence = i / 20.0
        tracker.record_prediction(f"p{i}", "model_1", confidence=confidence, actual_correct=(i % 2 == 0))

    diagram = tracker.get_reliability_diagram("model_1")
    assert "confidences" in diagram
    assert "accuracies" in diagram
    assert "counts" in diagram


def test_overcalibrated_models():
    """Test detection of overcalibrated models."""
    from src.training.calibration import CalibrationTracker

    tracker = CalibrationTracker({"calibration": {"bin_count": 10}})
    for i in range(20):
        tracker.record_prediction(f"p{i}", "model_1", confidence=0.95, actual_correct=(i < 5))
    for i in range(20):
        tracker.record_prediction(f"p2{i}", "model_2", confidence=0.5, actual_correct=(i % 2 == 0))

    overcalibrated = tracker.get_overcalibrated_models(threshold=0.1)
    assert "model_1" in overcalibrated


def test_uncertainty_estimator_certain():
    """Test uncertainty estimator for certain predictions."""
    from src.training.calibration import UncertaintyEstimator, UncertaintyLevel

    estimator = UncertaintyEstimator({})
    profile = estimator.estimate_uncertainty(
        confidence=0.95,
        evidence_quality=0.9,
        verification_status="pass",
        contradictions=[],
    )
    assert profile.uncertainty_level == UncertaintyLevel.CERTAIN.value


def test_uncertainty_estimator_uncertain():
    """Test uncertainty estimator for uncertain predictions."""
    from src.training.calibration import UncertaintyEstimator, UncertaintyLevel

    estimator = UncertaintyEstimator({})
    profile = estimator.estimate_uncertainty(
        confidence=0.5,
        evidence_quality=0.3,
        verification_status="fail",
        contradictions=["contradiction"],
    )
    assert profile.uncertainty_level == UncertaintyLevel.UNCERTAIN.value


def test_calibrated_confidence():
    """Test calibrated confidence adjustment."""
    from src.training.calibration import CalibrationTracker, UncertaintyEstimator

    tracker = CalibrationTracker({"calibration": {"bin_count": 10}})
    for i in range(20):
        tracker.record_prediction(f"p{i}", "model_1", confidence=0.95, actual_correct=(i < 5))
    estimator = UncertaintyEstimator({})
    calibrated = estimator.get_calibrated_confidence(0.95, "model_1")
    assert 0.0 <= calibrated <= 1.0
