"""Teacher distillation support for Prothynesis."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum


class DistillationMode(Enum):
    """Distillation modes."""
    HARD = "hard"  # Teacher final answers only
    SOFT = "soft"  # Teacher logits only
    MIXED = "mixed"  # Both hard and soft
    MULTI_TEACHER = "multi_teacher"  # Multiple teachers with weights


@dataclass
class TeacherConfig:
    """Configuration for a teacher model."""
    teacher_id: str
    weight: float = 1.0
    temperature: float = 2.0
    mode: str = "soft"
    specialties: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DistillationBatch:
    """Batch with teacher signals."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    teacher_logits: Optional[torch.Tensor] = None
    teacher_labels: Optional[torch.Tensor] = None
    teacher_weights: Optional[torch.Tensor] = None
    sample_difficulties: Optional[torch.Tensor] = None

    def to(self, device: torch.device, non_blocking: bool = True):
        self.input_ids = self.input_ids.to(device, non_blocking=non_blocking)
        self.labels = self.labels.to(device, non_blocking=non_blocking)
        if self.teacher_logits is not None:
            self.teacher_logits = self.teacher_logits.to(device, non_blocking=non_blocking)
        if self.teacher_labels is not None:
            self.teacher_labels = self.teacher_labels.to(device, non_blocking=non_blocking)
        if self.teacher_weights is not None:
            self.teacher_weights = self.teacher_weights.to(device, non_blocking=non_blocking)
        if self.sample_difficulties is not None:
            self.sample_difficulties = self.sample_difficulties.to(device, non_blocking=non_blocking)
        return self


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss combining:
    - Soft target loss (KL divergence with teacher logits)
    - Hard target loss (ground truth cross-entropy)
    - Difficulty-weighted sample loss
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        hard_weight: float = 0.5,
        soft_weight: float = 0.5,
        difficulty_scaling: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.difficulty_scaling = difficulty_scaling

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: Optional[torch.Tensor],
        labels: torch.Tensor,
        teacher_labels: Optional[torch.Tensor] = None,
        sample_difficulties: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: [batch, seq, vocab]
            teacher_logits: [batch, seq, vocab] or None
            labels: [batch, seq]
            teacher_labels: [batch, seq] or None
            sample_difficulties: [batch] or None
            
        Returns:
            Dict with total_loss, hard_loss, soft_loss, difficulty_loss
        """
        # Shift for causal LM
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Hard loss (standard CE)
        hard_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        )

        # Difficulty weighting
        if self.difficulty_scaling and sample_difficulties is not None:
            # Upscale hard examples
            difficulty_weights = 1.0 + sample_difficulties
            hard_loss = hard_loss * difficulty_weights.repeat_interleave(shift_logits.size(1))

        hard_loss = hard_loss.mean()

        # Soft loss (KL divergence with teacher)
        soft_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits is not None:
            teacher_logits_shifted = teacher_logits[..., :-1, :].contiguous()
            soft_targets = F.softmax(teacher_logits_shifted / self.temperature, dim=-1)
            student_soft = F.log_softmax(shift_logits / self.temperature, dim=-1)
            soft_loss = F.kl_div(student_soft, soft_targets, reduction="batchmean") * (self.temperature ** 2)

        # Hard labels from teacher (if provided)
        teacher_hard_loss = torch.tensor(0.0, device=student_logits.device)
        if teacher_labels is not None:
            teacher_labels_shifted = teacher_labels[..., 1:].contiguous()
            teacher_hard_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                teacher_labels_shifted.view(-1),
                ignore_index=-100,
            )

        # Combined loss
        total_loss = (
            self.hard_weight * hard_loss +
            self.soft_weight * soft_loss +
            self.alpha * teacher_hard_loss
        )

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "teacher_hard_loss": teacher_hard_loss,
        }


class MultiTeacherDistillationLoss(nn.Module):
    """
    Distillation from multiple teachers with per-teacher weights.
    
    Supports:
    - Soft target averaging
    - Hard target averaging
    - Per-teacher temperature
    - Specialty-based routing (some teachers are better at certain tasks)
    """

    def __init__(
        self,
        teachers: List[TeacherConfig],
        hard_weight: float = 0.3,
        soft_weight: float = 0.7,
        difficulty_scaling: bool = True,
    ):
        super().__init__()
        self.teachers = teachers
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.difficulty_scaling = difficulty_scaling

        # Validate weights
        total_weight = sum(t.weight for t in teachers)
        if total_weight > 0:
            for teacher in teachers:
                teacher.weight = teacher.weight / total_weight

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits_list: List[torch.Tensor],
        labels: torch.Tensor,
        sample_difficulties: Optional[torch.Tensor] = None,
        task_types: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-teacher distillation loss.
        
        Args:
            student_logits: [batch, seq, vocab]
            teacher_logits_list: List of [batch, seq, vocab]
            labels: [batch, seq]
            sample_difficulties: [batch] or None
            task_types: [batch] or None (for specialty routing)
            
        Returns:
            Dict with aggregated and per-teacher losses
        """
        device = student_logits.device
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Hard loss
        hard_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Aggregate soft loss across teachers
        soft_loss = torch.tensor(0.0, device=device)
        total_weight = 0.0

        for i, (teacher, t_logits) in enumerate(zip(self.teachers, teacher_logits_list)):
            if t_logits is None:
                continue

            t_logits_shifted = t_logits[..., :-1, :].contiguous()
            temperature = teacher.temperature
            weight = teacher.weight

            soft_targets = F.softmax(t_logits_shifted / temperature, dim=-1)
            student_soft = F.log_softmax(shift_logits / temperature, dim=-1)
            teacher_soft_loss = F.kl_div(student_soft, soft_targets, reduction="batchmean") * (temperature ** 2)

            # Specialty routing: boost weight if task matches teacher specialty
            if task_types is not None and teacher.specialties:
                # Simplified: boost if any specialty matches
                specialty_boost = 1.0
                weight = weight * specialty_boost

            soft_loss = soft_loss + weight * teacher_soft_loss
            total_weight += weight

        if total_weight > 0:
            soft_loss = soft_loss / total_weight

        # Combined
        total_loss = self.hard_weight * hard_loss + self.soft_weight * soft_loss

        return {
            "total_loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
        }


def compute_distillation_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> Dict[str, float]:
    """Compute distillation quality metrics."""
    with torch.no_grad():
        # Agreement rate (top-1)
        student_preds = student_logits.argmax(dim=-1)
        teacher_preds = teacher_logits.argmax(dim=-1)
        agreement = (student_preds == teacher_preds).float().mean().item()

        # KL divergence
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        kl_div = F.kl_div(
            student_probs.log(), teacher_probs, reduction="batchmean"
        ).item()

        # Top-k agreement
        for k in [5, 10, 20]:
            student_topk = student_logits.topk(k, dim=-1).indices
            teacher_topk = teacher_logits.topk(k, dim=-1).indices
            agreement_k = sum(s.tolist() == t.tolist() for s, t in zip(student_topk, teacher_topk))
            agreement_k = agreement_k / student_logits.size(0)

    return {
        "distillation_agreement_top1": agreement,
        "distillation_kl_div": kl_div,
    }
