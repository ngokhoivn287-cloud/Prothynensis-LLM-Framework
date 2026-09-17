"""Expert FFN module (SwiGLU)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .attention import get_norm_layer


class SwiGLU(nn.Module):
    """SwiGLU activation: x * SiLU(gate)"""
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * F.silu(gate)


class ExpertFFN(nn.Module):
    """Expert Feed-Forward Network with SwiGLU activation.
    
    Architecture:
    - Input norm
    - Gate projection: hidden_dim -> expert_hidden_dim
    - Up projection: hidden_dim -> expert_hidden_dim
    - SwiGLU activation
    - Down projection: expert_hidden_dim -> hidden_dim
    - Optional output norm
    """
    
    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        activation: str = "silu",
        norm_type: str = "rmsnorm",
        bias: bool = False,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.expert_hidden_dim = expert_hidden_dim
        self.dropout = dropout
        
        # Input normalization
        self.input_norm = get_norm_layer(norm_type, hidden_dim, norm_eps)
        
        # SwiGLU projections (gate + up)
        self.gate_proj = nn.Linear(hidden_dim, expert_hidden_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, expert_hidden_dim, bias=bias)
        
        # Activation
        if activation == "silu":
            self.act_fn = F.silu
        elif activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "relu":
            self.act_fn = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Down projection
        self.down_proj = nn.Linear(expert_hidden_dim, hidden_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output normalization (optional, for deep expert stacks)
        self.output_norm = get_norm_layer(norm_type, hidden_dim, norm_eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim] or [batch_size * seq_len, hidden_dim]
        
        Returns:
            output: Same shape as input
        """
        # Save residual
        residual = x
        
        # Input norm
        x = self.input_norm(x)
        
        # Gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU
        x = up * self.act_fn(gate)
        
        # Dropout
        x = self.dropout_layer(x)
        
        # Down projection
        x = self.down_proj(x)
        
        # Output norm
        x = self.output_norm(x)
        
        # Residual
        return residual + x
    
    def get_parameter_count(self) -> int:
        """Get number of parameters in this expert."""
        count = 0
        count += sum(p.numel() for p in self.input_norm.parameters())
        count += self.gate_proj.weight.numel()
        if self.gate_proj.bias is not None:
            count += self.gate_proj.bias.numel()
        count += self.up_proj.weight.numel()
        if self.up_proj.bias is not None:
            count += self.up_proj.bias.numel()
        count += self.down_proj.weight.numel()
        if self.down_proj.bias is not None:
            count += self.down_proj.bias.numel()
        count += sum(p.numel() for p in self.output_norm.parameters())
        return count


class Expert(nn.Module):
    """Wrapper for expert FFN that can be used standalone or in MoE."""
    
    def __init__(
        self,
        hidden_dim: int,
        expert_hidden_dim: int,
        expert_id: int,
        activation: str = "silu",
        norm_type: str = "rmsnorm",
        bias: bool = False,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.expert_id = expert_id
        self.ffn = ExpertFFN(
            hidden_dim=hidden_dim,
            expert_hidden_dim=expert_hidden_dim,
            activation=activation,
            norm_type=norm_type,
            bias=bias,
            dropout=dropout,
            norm_eps=norm_eps,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)
    
    def get_parameter_count(self) -> int:
        return self.ffn.get_parameter_count()


def create_expert(
    hidden_dim: int,
    expert_hidden_dim: int,
    expert_id: int,
    config: dict,
) -> Expert:
    """Factory function to create expert from config."""
    return Expert(
        hidden_dim=hidden_dim,
        expert_hidden_dim=expert_hidden_dim,
        expert_id=expert_id,
        activation=config.get("expert_activation", "silu"),
        norm_type=config.get("expert_norm", "rmsnorm"),
        bias=config.get("expert_bias", False),
        dropout=float(config.get("expert_dropout", 0.0)),
        norm_eps=float(config.get("norm_eps", 1e-5)),
    )