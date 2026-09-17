"""Cross-Model Fusion Layer for MoMMs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    """Configuration for cross-model fusion."""
    fusion_type: str = "attention"  # "weighted_sum", "attention", "cross_attention", "gated"
    hidden_dim: int = 768
    num_models: int = 32
    top_k: int = 2
    fusion_hidden_dim: Optional[int] = None
    num_fusion_layers: int = 2
    num_fusion_heads: int = 8
    fusion_head_dim: int = 64
    dropout: float = 0.0
    use_residual: bool = True
    temperature: float = 1.0
    
    def to_dict(self) -> dict:
        return {
            "fusion_type": self.fusion_type,
            "hidden_dim": self.hidden_dim,
            "num_models": self.num_models,
            "top_k": self.top_k,
            "fusion_hidden_dim": self.fusion_hidden_dim,
            "num_fusion_layers": self.num_fusion_layers,
            "num_fusion_heads": self.num_fusion_heads,
            "fusion_head_dim": self.fusion_head_dim,
            "dropout": self.dropout,
            "use_residual": self.use_residual,
            "temperature": self.temperature,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "FusionConfig":
        return cls(**d)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CrossModelAttention(nn.Module):
    """Cross-model attention for fusing outputs from multiple models."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_models: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        
        # Project each model's output to shared space
        self.model_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_models)
        ])
        
        # Multi-head attention over model outputs
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        model_outputs: List[torch.Tensor],  # List of [batch, seq, hidden]
        model_weights: Optional[torch.Tensor] = None,  # [batch, seq, num_models] or [num_models]
    ) -> torch.Tensor:
        """
        Fuse outputs from multiple models using cross-model attention.
        
        Args:
            model_outputs: List of [batch, seq, hidden] from each model
            model_weights: Optional routing weights [batch, seq, num_models] or [num_models]
        
        Returns:
            Fused output: [batch, seq, hidden]
        """
        batch_size = model_outputs[0].size(0)
        seq_len = model_outputs[0].size(1)
        num_models = len(model_outputs)
        hidden_dim = model_outputs[0].size(-1)
        
        # Project each model's output
        projected = []
        for i, out in enumerate(model_outputs):
            projected.append(self.model_projections[i](out))
        
        # Stack: [num_models, batch, seq, hidden]
        stacked = torch.stack(projected, dim=0)  # [num_models, batch, seq, hidden]
        
        # Reshape for attention: [batch, seq, num_models, hidden] -> [batch*seq, num_models, hidden]
        stacked = stacked.permute(1, 2, 0, 3).reshape(batch_size * seq_len, num_models, -1)
        
        # Self-attention over models
        # Q: [batch*seq, num_models, hidden] -> [batch*seq, num_models, heads, head_dim]
        q = self.q_proj(stacked).view(-1, num_models, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(stacked).view(-1, num_models, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(stacked).view(-1, num_models, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=0.0, training=self.training)
        
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size * seq_len, num_models, self.num_heads * self.head_dim)
        attn_output = self.out_proj(attn_output)
        
        # Reshape back: [batch*seq, num_models, hidden] -> [batch, seq, num_models, hidden]
        attn_output = attn_output.view(batch_size, seq_len, num_models, -1)
        
        # Weighted sum if weights provided
        if model_weights is not None:
            if model_weights.dim() == 1:
                # [num_models] -> [1, 1, num_models, 1]
                weights = model_weights.view(1, 1, num_models, 1)
            elif model_weights.dim() == 2:
                # [batch, num_models] -> [batch, 1, num_models, 1]
                weights = model_weights.unsqueeze(1).unsqueeze(-1)
            elif model_weights.dim() == 3:
                # [batch, seq, num_models] -> [batch, seq, num_models, 1]
                weights = model_weights.unsqueeze(-1)
            else:
                weights = None
            
            if weights is not None:
                fused = (attn_output * weights).sum(dim=2)
            else:
                fused = attn_output.mean(dim=2)
        else:
            fused = attn_output.mean(dim=2)
        
        return fused


class WeightedSumFusion(nn.Module):
    """Simple weighted sum fusion."""
    
    def __init__(self, hidden_dim: int, num_models: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        
        # Learnable model weights
        self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
        # Optional projection for each model
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_models)
        ])
    
    def forward(
        self,
        model_outputs: List[torch.Tensor],
        model_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            model_outputs: List of [batch, seq, hidden]
            model_weights: Optional [num_models] or [batch, seq, num_models]
        """
        batch_size = model_outputs[0].size(0)
        seq_len = model_outputs[0].size(1)
        
        projected = [self.projections[i](out) for i, out in enumerate(model_outputs)]
        stacked = torch.stack(projected, dim=2)  # [batch, seq, num_models, hidden]
        
        if model_weights is not None:
            weights = model_weights
            if weights.dim() == 1:
                weights = weights.view(1, 1, -1)
            elif weights.dim() == 2:
                weights = weights.unsqueeze(1)
        else:
            weights = F.softmax(self.model_weights, dim=0).view(1, 1, -1)
        
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return fused


class GatedFusion(nn.Module):
    """Gated fusion with learnable gates."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_models: int,
        gate_hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        self.gate_hidden_dim = gate_hidden_dim or hidden_dim
        
        # Gate network
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * num_models, self.gate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.gate_hidden_dim, num_models),
        )
        
        # Optional projections
        self.projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_models)
        ])
    
    def forward(
        self,
        model_outputs: List[torch.Tensor],
        model_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = model_outputs[0].size(0)
        seq_len = model_outputs[0].size(1)
        hidden_dim = model_outputs[0].size(-1)
        
        # Project each output
        projected = [self.projections[i](out) for i, out in enumerate(model_outputs)]
        stacked = torch.stack(projected, dim=2)  # [batch, seq, num_models, hidden]
        
        # Concatenate for gate
        concat = stacked.view(-1, self.num_models * self.hidden_dim)
        
        # Compute gates
        gates = self.gate_network(concat)  # [batch*seq, num_models]
        gates = F.softmax(gates, dim=-1).view(-1, self.num_models, 1)  # [batch*seq, num_models, 1]
        
        # Apply gates
        fused = (stacked * gates).sum(dim=2)
        
        return fused


class FusionLayer(nn.Module):
    """
    Cross-Model Fusion Layer.
    
    Fuses outputs from multiple models using configurable fusion strategies.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        self.config = config
        self.num_models = config.num_models
        self.hidden_dim = config.hidden_dim
        self.fusion_type = config.fusion_type
        
        if config.fusion_hidden_dim is None:
            fusion_hidden = config.hidden_dim
        else:
            fusion_hidden = config.fusion_hidden_dim
        
        # Model-specific projections
        self.model_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
            for _ in range(config.num_models)
        ])
        
        # Create fusion module
        if config.fusion_type == "weighted_sum":
            self.fusion = WeightedSumFusion(config.hidden_dim, config.num_models)
        elif config.fusion_type == "attention":
            self.fusion = CrossModelAttention(
                hidden_dim=config.hidden_dim,
                num_models=config.num_models,
                num_heads=config.num_fusion_heads,
                head_dim=config.fusion_head_dim,
                dropout=config.dropout,
            )
        elif config.fusion_type == "cross_attention":
            self.fusion = CrossModelAttention(
                hidden_dim=config.hidden_dim,
                num_models=config.num_models,
                num_heads=config.num_fusion_heads,
                head_dim=config.fusion_head_dim,
                dropout=config.dropout,
            )
        elif config.fusion_type == "gated":
            self.fusion = GatedFusion(
                hidden_dim=config.hidden_dim,
                num_models=config.num_models,
                gate_hidden_dim=config.fusion_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")
        
        # Final norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)
        
        # Temperature for routing
        self.temperature = config.temperature
    
    def forward(
        self,
        model_outputs: List[torch.Tensor],  # List of [batch, seq, hidden]
        routing_weights: Optional[torch.Tensor] = None,  # [batch, seq, num_models] or [num_models]
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse outputs from multiple models.
        
        Args:
            model_outputs: List of [batch, seq, hidden] from each model
            routing_weights: Optional routing weights [batch, seq, num_models] or [num_models]
            return_weights: Whether to return the computed weights
        
        Returns:
            Fused output [batch, seq, hidden] and optionally weights
        """
        batch_size = model_outputs[0].size(0)
        seq_len = model_outputs[0].size(1)
        
        # Project each model's output
        projected = [self.model_projections[i](out) for i, out in enumerate(model_outputs)]
        
        # Apply fusion
        if self.config.fusion_type in ("weighted_sum", "gated"):
            fused = self.fusion(model_outputs, model_weights=routing_weights)
        else:
            fused = self.fusion(model_outputs, routing_weights)
        
        # Apply final norm
        fused = self.final_norm(fused)
        
        # Residual connection if enabled
        if self.config.use_residual and len(model_outputs) > 0:
            # Add residual from first model (or average)
            residual = model_outputs[0]
            fused = fused + residual
        
        return fused
    
    def get_parameter_count(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "fusion": total}


def create_fusion(config: FusionConfig | Dict[str, Any]) -> FusionLayer:
    """Factory function to create fusion layer."""
    if isinstance(config, dict):
        config = FusionConfig.from_dict(config)
    return FusionLayer(config)