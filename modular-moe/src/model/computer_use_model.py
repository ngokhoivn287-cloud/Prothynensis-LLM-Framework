"""Computer-Use model family for Ultra 3 / Trinity 3 MrTP."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_model import VisionTransformerBackbone, VisionModelConfig, PatchEmbedding
from ..momm.model import RMSNorm
from ..utils.config import convert_config_types


@dataclass
class ComputerUseModelConfig:
    model_id: str = "computer_use_model"
    vocab_size: int = 50304
    max_seq_len: int = 2048
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    ffn_hidden_dim: int = 2048
    activation: str = "silu"
    norm_type: str = "rmsnorm"
    tie_embeddings: bool = True
    dropout: float = 0.0

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    vision_hidden_dim: int = 768
    vision_num_layers: int = 12
    vision_num_heads: int = 12
    vision_head_dim: int = 64
    use_flash_attn: bool = False
    gradient_checkpointing: bool = False

    num_action_types: int = 10
    max_actions_per_step: int = 5
    action_hidden_dim: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ComputerUseModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ActionHead(nn.Module):
    """Predicts computer-use actions from visual features."""

    def __init__(self, hidden_dim: int, num_action_types: int, max_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_action_types = num_action_types
        self.max_actions = max_actions
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.action_type = nn.Linear(hidden_dim, num_action_types)
        self.action_params = nn.Linear(hidden_dim, max_actions * 4)
        self.confidence = nn.Linear(hidden_dim, max_actions)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.gelu(self.proj(x))
        action_type = self.action_type(x)
        action_params = self.action_params(x).view(-1, self.max_actions, 4)
        confidence = torch.sigmoid(self.confidence(x))
        return {
            "action_type": action_type,
            "action_params": action_params,
            "confidence": confidence,
        }


class ComputerUseModel(nn.Module):
    """Computer-use model with vision backbone and action head."""

    def __init__(self, config: ComputerUseModelConfig):
        super().__init__()
        self.config = config
        self.vision = VisionTransformerBackbone(
            VisionModelConfig(
                model_id=config.model_id,
                vocab_size=config.vocab_size,
                max_seq_len=config.max_seq_len,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                activation=config.activation,
                norm_type=config.norm_type,
                tie_embeddings=config.tie_embeddings,
                dropout=config.dropout,
                image_size=config.image_size,
                patch_size=config.patch_size,
                num_channels=config.num_channels,
                vision_hidden_dim=config.vision_hidden_dim,
                vision_num_layers=config.vision_num_layers,
                vision_num_heads=config.vision_num_heads,
                vision_head_dim=config.vision_head_dim,
            )
        )
        self.proj = nn.Linear(config.vision_hidden_dim, config.hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                RMSNorm(config.hidden_dim),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            for _ in range(config.num_layers)
        ])
        self.ln_final = RMSNorm(config.hidden_dim)
        self.action_head = ActionHead(
            hidden_dim=config.hidden_dim,
            num_action_types=config.num_action_types,
            max_actions=config.max_actions_per_step,
        )
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.vision(images)
        x = self.proj(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.ln_final(x)
        actions = self.action_head(x)
        logits = self.lm_head(x)
        return {"logits": logits, "actions": actions}


def create_computer_use_model(config: ComputerUseModelConfig | Dict[str, Any]) -> ComputerUseModel:
    if isinstance(config, dict):
        config = ComputerUseModelConfig.from_dict(convert_config_types(config))
    return ComputerUseModel(config)


def count_computer_use_model_parameters(model: ComputerUseModel) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
