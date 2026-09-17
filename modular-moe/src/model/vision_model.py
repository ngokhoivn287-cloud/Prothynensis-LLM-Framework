"""Vision model family for Ultra 3 / Trinity 3 MrTP."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..momm.model import RMSNorm
from ..utils.config import convert_config_types


@dataclass
class VisionModelConfig:
    model_id: str = "vision_model"
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VisionModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PatchEmbedding(nn.Module):
    """Image-to-patch embedding."""

    def __init__(self, image_size: int = 224, patch_size: int = 16, num_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(num_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformerBackbone(nn.Module):
    """Vision transformer backbone for image understanding."""

    def __init__(self, config: VisionModelConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embed_dim=config.vision_hidden_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.vision_hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.vision_hidden_dim))
        self.pos_drop = nn.Dropout(p=config.dropout)
        self.ln_final = RMSNorm(config.vision_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.ln_final(x)
        return x


class VisionReasoningModel(nn.Module):
    """Vision model with transformer reasoning head."""

    def __init__(self, config: VisionModelConfig):
        super().__init__()
        self.config = config
        self.vision = VisionTransformerBackbone(config)
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
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.vision(images)
        x = self.proj(x)
        for layer in self.layers:
            x = x + layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


def create_vision_model(config: VisionModelConfig | Dict[str, Any]) -> VisionReasoningModel:
    if isinstance(config, dict):
        config = VisionModelConfig.from_dict(convert_config_types(config))
    return VisionReasoningModel(config)


def count_vision_model_parameters(model: VisionReasoningModel) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
