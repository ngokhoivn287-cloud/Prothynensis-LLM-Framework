"""Tests for recursive verification and meta-verifier."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_recursive_verification_converges():
    """Test recursive verification converges on pass."""
    from src.training.verification import RecursiveVerifier, VerificationOutcome

    verifier = RecursiveVerifier({"recursive_verification": {"max_rounds": 3, "convergence_threshold": 0.9}})

    def verification_fn(content: str, verifier_id: str):
        return type("Result", (), {
            "outcome": VerificationOutcome.PASS.value,
            "confidence": 0.95,
            "revision_suggestions": [],
        })()

    state = verifier.verify("candidate_1", "answer", "verifier_1", verification_fn)
    assert state.converged is True
    assert state.final_outcome == VerificationOutcome.PASS.value
    assert state.round == 1


def test_recursive_verification_needs_revision():
    """Test recursive verification with revision."""
    from src.training.verification import RecursiveVerifier, VerificationOutcome

    verifier = RecursiveVerifier({"recursive_verification": {"max_rounds": 3}})
    call_count = 0

    def verification_fn(content: str, verifier_id: str):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            return type("Result", (), {
                "outcome": VerificationOutcome.NEEDS_REVISION.value,
                "confidence": 0.5,
                "revision_suggestions": ["fix assumption"],
            })()
        return type("Result", (), {
            "outcome": VerificationOutcome.PASS.value,
            "confidence": 0.9,
            "revision_suggestions": [],
        })()

    state = verifier.verify("candidate_1", "answer", "verifier_1", verification_fn)
    assert state.round == 2
    assert len(state.revisions) == 1
    assert state.final_outcome == VerificationOutcome.PASS.value


def test_recursive_verification_max_rounds():
    """Test recursive verification stops at max rounds."""
    from src.training.verification import RecursiveVerifier, VerificationOutcome

    verifier = RecursiveVerifier({"recursive_verification": {"max_rounds": 2}})

    def verification_fn(content: str, verifier_id: str):
        return type("Result", (), {
            "outcome": VerificationOutcome.NEEDS_REVISION.value,
            "confidence": 0.5,
            "revision_suggestions": ["fix"],
        })()

    state = verifier.verify("candidate_1", "answer", "verifier_1", verification_fn)
    assert state.round == 2
    assert state.final_outcome == VerificationOutcome.UNCERTAIN.value


def test_meta_verifier_insufficient_evidence():
    """Test meta-verifier rejects insufficient evidence."""
    from src.training.verification import MetaVerifier, VerificationOutcome

    meta = MetaVerifier({"meta_verifier": {"min_evidence_quality": 0.8}})

    def meta_fn(content, evidence, verifications):
        return type("Result", (), {
            "outcome": VerificationOutcome.PASS.value,
            "confidence": 0.9,
            "evidence": evidence,
        })()

    result = meta.meta_verify("c1", "answer", ["e1"], [], "meta_1", meta_fn)
    assert result.outcome == VerificationOutcome.UNCERTAIN.value
    assert "insufficient_evidence_quality" in result.metadata.get("reason", "")


def test_meta_verifier_inconsistent_verifications():
    """Test meta-verifier detects inconsistent verifications."""
    from src.training.verification import MetaVerifier, VerificationOutcome, VerificationResult

    meta = MetaVerifier({"meta_verifier": {"consistency_threshold": 0.8}})
    verifications = [
        VerificationResult("v1", "c1", "v1", VerificationOutcome.PASS.value, 0.9, metadata={}),
        VerificationResult("v2", "c1", "v2", VerificationOutcome.FAIL.value, 0.8, metadata={}),
        VerificationResult("v3", "c1", "v3", VerificationOutcome.FAIL.value, 0.85, metadata={}),
        VerificationResult("v4", "c1", "v4", VerificationOutcome.PASS.value, 0.9, metadata={}),
        VerificationResult("v5", "c1", "v5", VerificationOutcome.PASS.value, 0.88, metadata={}),
    ]

    def meta_fn(content, evidence, verifications):
        return VerificationResult(
            verification_id="meta_c1_test",
            candidate_id="c1",
            verifier_id="meta_1",
            outcome=VerificationOutcome.PASS.value,
            confidence=0.9,
            evidence=evidence,
            metadata={},
        )

    result = meta.meta_verify("c1", "answer", ["e1", "e2", "e3", "e4", "e5"], verifications, "meta_1", meta_fn)
    assert result.outcome == VerificationOutcome.NEEDS_RECHECK.value


def test_meta_verifier_passes():
    """Test meta-verifier passes with consistent verifications."""
    from src.training.verification import MetaVerifier, VerificationOutcome, VerificationResult

    meta = MetaVerifier({"meta_verifier": {"min_evidence_quality": 0.1, "consistency_threshold": 0.5}})
    verifications = [
        VerificationResult("v1", "c1", "v1", VerificationOutcome.PASS.value, 0.9, metadata={}),
        VerificationResult("v2", "c1", "v2", VerificationOutcome.PASS.value, 0.85, metadata={}),
    ]

    def meta_fn(content, evidence, verifications):
        return VerificationResult(
            verification_id="meta_c1_test",
            candidate_id="c1",
            verifier_id="meta_1",
            outcome=VerificationOutcome.PASS.value,
            confidence=0.95,
            evidence=evidence,
            metadata={},
        )

    result = meta.meta_verify("c1", "answer", ["e1", "e2"], verifications, "meta_1", meta_fn)
    assert result.outcome == VerificationOutcome.PASS.value
