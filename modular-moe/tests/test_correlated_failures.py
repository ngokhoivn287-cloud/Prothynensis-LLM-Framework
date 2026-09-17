"""Tests for correlated failure detection."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_correlated_failure_detector_records_failures():
    """Test that correlated failure detector records failures."""
    from src.training.correlated_failures import CorrelatedFailureDetector

    detector = CorrelatedFailureDetector({
        "correlated_failures": {
            "correlation_threshold": 0.5,
            "min_failure_count": 2,
            "time_window": 3600.0,
        }
    })
    for i in range(3):
        detector.record_failure(
            model_id=f"model_{i}",
            task_id="task_1",
            failure_category="HIGH_VAL_LOSS",
            description="High validation loss",
            confidence=0.3,
        )

    groups = detector.get_correlated_failures()
    assert len(groups) >= 1
    assert groups[0].failure_category == "HIGH_VAL_LOSS"


def test_correlated_failure_contamination_risk():
    """Test contamination risk assessment."""
    from src.training.correlated_failures import CorrelatedFailureDetector

    detector = CorrelatedFailureDetector({
        "correlated_failures": {
            "correlation_threshold": 0.5,
            "min_failure_count": 2,
            "time_window": 3600.0,
        }
    })
    for i in range(3):
        detector.record_failure(
            model_id=f"model_{i}",
            task_id="task_1",
            failure_category="HIGH_VAL_LOSS",
            description="High validation loss",
        )

    risk = detector.get_contamination_risk(["model_0", "model_1", "model_2"])
    assert risk["correlated_groups"] >= 1


def test_correlated_failure_statistics():
    """Test failure statistics."""
    from src.training.correlated_failures import CorrelatedFailureDetector

    detector = CorrelatedFailureDetector({
        "correlated_failures": {
            "min_failure_count": 2,
            "time_window": 3600.0,
        }
    })
    for i in range(3):
        detector.record_failure(
            model_id=f"model_{i}",
            task_id="task_1",
            failure_category="HIGH_VAL_LOSS",
            description="High validation loss",
        )
    for i in range(2):
        detector.record_failure(
            model_id=f"model_{i}",
            task_id="task_2",
            failure_category="OVERFIT",
            description="Overfitting detected",
        )

    stats = detector.get_statistics()
    assert stats["total_events"] == 5
    assert stats["correlated_groups"] >= 1
