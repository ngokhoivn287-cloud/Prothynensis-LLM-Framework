"""Tests for teacher distillation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_distillation_loss_forward():
    """Test distillation loss computation."""
    import torch
    from src.training.distillation import DistillationLoss

    loss_fn = DistillationLoss(temperature=2.0, alpha=0.5)

    batch_size, seq_len, vocab_size = 2, 8, 100
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    result = loss_fn(student_logits, teacher_logits, labels)

    assert "total_loss" in result
    assert "hard_loss" in result
    assert "soft_loss" in result
    assert result["total_loss"].item() > 0
    print("  PASS: Distillation loss computed")


def test_distillation_loss_without_teacher():
    """Test distillation loss with no teacher logits."""
    import torch
    from src.training.distillation import DistillationLoss

    loss_fn = DistillationLoss()
    batch_size, seq_len, vocab_size = 2, 8, 100
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    result = loss_fn(student_logits, None, labels)
    assert result["total_loss"].item() > 0
    assert result["soft_loss"].item() == 0.0
    print("  PASS: Distillation without teacher falls back to hard loss")


def test_distillation_with_difficulty_weighting():
    """Test difficulty-weighted distillation."""
    import torch
    from src.training.distillation import DistillationLoss

    loss_fn = DistillationLoss(difficulty_scaling=True)
    batch_size, seq_len, vocab_size = 2, 8, 100
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    difficulties = torch.tensor([0.5, 2.0])

    result = loss_fn(student_logits, teacher_logits, labels, sample_difficulties=difficulties)
    assert result["total_loss"].item() > 0
    print("  PASS: Difficulty-weighted distillation works")


def test_multi_teacher_distillation():
    """Test multi-teacher distillation."""
    import torch
    from src.training.distillation import MultiTeacherDistillationLoss, TeacherConfig

    teachers = [
        TeacherConfig(teacher_id="teacher_a", weight=0.5, temperature=2.0),
        TeacherConfig(teacher_id="teacher_b", weight=0.3, temperature=2.0),
        TeacherConfig(teacher_id="teacher_c", weight=0.2, temperature=2.0),
    ]

    loss_fn = MultiTeacherDistillationLoss(teachers=teachers)
    batch_size, seq_len, vocab_size = 2, 8, 100
    student_logits = torch.randn(batch_size, seq_len, vocab_size)
    teacher_logits_list = [torch.randn(batch_size, seq_len, vocab_size) for _ in teachers]
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    result = loss_fn(student_logits, teacher_logits_list, labels)
    assert result["total_loss"].item() > 0
    print("  PASS: Multi-teacher distillation works")


def test_teacher_config_serialization():
    """Test teacher config serialization."""
    from src.training.distillation import TeacherConfig

    config = TeacherConfig(
        teacher_id="teacher_1",
        weight=0.8,
        temperature=3.0,
        specialties=["reasoning", "math"],
    )

    data = config.to_dict()
    assert data["teacher_id"] == "teacher_1"
    assert data["weight"] == 0.8
    assert data["specialties"] == ["reasoning", "math"]
    print("  PASS: Teacher config serialization works")


def test_distillation_metrics():
    """Test distillation quality metrics."""
    import torch
    from src.training.distillation import compute_distillation_metrics

    vocab_size = 100
    student_logits = torch.randn(4, 8, vocab_size)
    teacher_logits = torch.randn(4, 8, vocab_size)

    metrics = compute_distillation_metrics(student_logits, teacher_logits)
    assert "distillation_agreement_top1" in metrics
    assert "distillation_kl_div" in metrics
    assert 0.0 <= metrics["distillation_agreement_top1"] <= 1.0
    print("  PASS: Distillation metrics computed")


def test_distillation_modes():
    """Test distillation mode enum."""
    from src.training.distillation import DistillationMode

    assert DistillationMode.HARD.value == "hard"
    assert DistillationMode.SOFT.value == "soft"
    assert DistillationMode.MIXED.value == "mixed"
    assert DistillationMode.MULTI_TEACHER.value == "multi_teacher"
    print("  PASS: Distillation modes defined")


def run_all_tests():
    """Run all distillation tests."""
    tests = [
        ("Distillation Loss Forward", test_distillation_loss_forward),
        ("Distillation Without Teacher", test_distillation_loss_without_teacher),
        ("Difficulty Weighting", test_distillation_with_difficulty_weighting),
        ("Multi-Teacher Distillation", test_multi_teacher_distillation),
        ("Teacher Config Serialization", test_teacher_config_serialization),
        ("Distillation Metrics", test_distillation_metrics),
        ("Distillation Modes", test_distillation_modes),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
