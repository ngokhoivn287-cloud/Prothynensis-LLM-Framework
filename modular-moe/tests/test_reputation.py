"""Tests for reputation and contribution tracking."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_reputation_tracks_accuracy():
    """Test that reputation tracks accuracy correctly."""
    from src.training.reputation import ReputationStore, ModelReputation

    store = ReputationStore()
    rep = store.get_or_create("model_1")
    for _ in range(2):
        rep.record_global_attempt(success=True, confidence=0.9, actual_correct=True)
    for _ in range(3):
        rep.record_global_attempt(success=False, confidence=0.8, actual_correct=False)
    rep.update_tier()
    store.update_reputation(rep)

    assert rep.global_attempts == 5
    assert abs(rep.global_accuracy - 0.4) < 0.01
    assert rep.tier == "competent"


def test_reputation_tracks_calibration():
    """Test that reputation tracks calibration error."""
    from src.training.reputation import ModelReputation

    rep = ModelReputation(model_id="model_1")
    for _ in range(10):
        rep.record_global_attempt(success=True, confidence=0.95, actual_correct=True)
    for _ in range(10):
        rep.record_global_attempt(success=False, confidence=0.9, actual_correct=False)

    assert rep.calibration_error > 0.0
    assert rep.reliability_score < 1.0


def test_contribution_ledger():
    """Test contribution ledger tracks model usefulness."""
    from src.training.reputation import ContributionLedger

    ledger = ContributionLedger(model_id="model_1")
    ledger.record_participation(contributed=True)
    ledger.record_participation(contributed=False)
    ledger.record_verification(correct=True)
    ledger.record_critique(helpful=True)
    ledger.record_tool_call(successful=True)

    assert ledger.tasks_participated == 2
    assert ledger.contribution_rate == 0.5
    assert ledger.verification_accuracy == 1.0


def test_reputation_persist_and_load(tmp_path):
    """Test reputation persistence."""
    from src.training.reputation import ReputationStore, ModelReputation

    store = ReputationStore(storage_path=str(tmp_path / "reputation.json"))
    rep = store.get_or_create("model_1")
    rep.record_global_attempt(success=True, confidence=0.9, actual_correct=True)
    store.update_reputation(rep)

    store2 = ReputationStore(storage_path=str(tmp_path / "reputation.json"))
    store2.load()
    rep2 = store2.get_reputation("model_1")
    assert rep2 is not None
    assert rep2.global_attempts == 1


def test_domain_reputation():
    """Test per-domain reputation tracking."""
    from src.training.reputation import ModelReputation

    rep = ModelReputation(model_id="model_1")
    rep.record_domain_attempt("math", success=True, confidence=0.9, actual_correct=True)
    rep.record_domain_attempt("math", success=False, confidence=0.8, actual_correct=False)
    rep.record_domain_attempt("coding", success=True, confidence=0.85, actual_correct=True)

    assert rep.get_domain_accuracy("math") == 0.5
    assert rep.get_domain_accuracy("coding") == 1.0


def test_get_most_reliable():
    """Test getting most reliable models."""
    from src.training.reputation import ReputationStore

    store = ReputationStore()
    for i in range(10):
        rep = store.get_or_create(f"model_{i}")
        successes = 5 if i >= 7 else (3 if i >= 4 else 0)
        for _ in range(5):
            actual_success = successes > 0
            rep.record_global_attempt(success=actual_success, confidence=0.8, actual_correct=actual_success)
            successes -= 1
        rep.update_tier()
        store.update_reputation(rep)

    reliable = store.get_most_reliable(min_attempts=5)
    assert len(reliable) > 0
    top_ids = [r.model_id for r in reliable[:3]]
    assert "model_9" in top_ids
