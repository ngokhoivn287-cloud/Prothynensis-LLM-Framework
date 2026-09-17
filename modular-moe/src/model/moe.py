"""Mixture-of-Experts layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .expert import Expert, ExpertFFN
from .router import Router, create_router
from .attention import get_norm_layer


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with Top-K routing.
    
    Replaces standard FFN in Transformer block.
    Supports:
    - Top-K routing with learned router
    - Shared experts across layers (same experts used in all MoE layers)
    - Load balancing auxiliary loss
    - Capacity factor (optional)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        expert_activation: str = "silu",
        expert_norm: str = "rmsnorm",
        expert_bias: bool = False,
        expert_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        router_hidden_dim: Optional[int] = None,
        router_bias: bool = False,
        router_dropout: float = 0.0,
        capacity_factor: float = 1.0,
        drop_tokens: bool = False,
        router_config: Optional[dict] = None,
        shared_experts: Optional[nn.ModuleList] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
        # Input normalization (shared)
        self.input_norm = get_norm_layer(expert_norm, hidden_dim, norm_eps)
        
        # Experts - either shared (passed in) or created locally
        if shared_experts is not None:
            self.experts = shared_experts
            self._owns_experts = False
        else:
            # Create local experts (for backward compatibility / single MoE layer)
            self.experts = nn.ModuleList([
                ExpertFFN(
                    hidden_dim=hidden_dim,
                    expert_hidden_dim=expert_hidden_dim,
                    activation=expert_activation,
                    norm_type=expert_norm,
                    bias=expert_bias,
                    dropout=expert_dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(num_experts)
            ])
            self._owns_experts = True
        
        # Router
        router_cfg = router_config or {}
        self.router = create_router(hidden_dim, num_experts, router_cfg)
        
        # Output normalization (optional)
        self.output_norm = get_norm_layer(expert_norm, hidden_dim, norm_eps)
        
        # Routing statistics tracking
        self.register_buffer("total_tokens", torch.zeros(1, dtype=torch.long))
        self.register_buffer("expert_token_counts", torch.zeros(num_experts, dtype=torch.long))
    
    def owns_experts(self) -> bool:
        """Return True if this layer owns its experts (not shared)."""
        return self._owns_experts
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
            return_aux_loss: Whether to compute load balancing loss
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            aux_loss: Load balancing loss (or None)
            routing_stats: Dictionary of routing statistics
        """
        batch_size, seq_len, _ = x.shape
        
        # Save residual
        residual = x
        
        # Input norm
        x = self.input_norm(x)
        
        # Flatten for routing: [batch*seq, hidden_dim]
        x_flat = x.view(-1, self.hidden_dim)
        
        # Handle attention mask
        if attention_mask is not None:
            # Flatten mask and filter valid tokens
            valid_mask = attention_mask.view(-1).bool()
            x_flat = x_flat[valid_mask]
        
        num_tokens = x_flat.size(0)
        
        if num_tokens == 0:
            # No valid tokens, return residual
            return residual, None, {}
        
        # Routing
        routing_weights, expert_indices, router_logits, aux_loss = self.router(
            x_flat, top_k=self.top_k, return_aux_loss=return_aux_loss
        )
        
        # Update statistics
        if self.training:
            self._update_stats(expert_indices)
        
        # Dispatch to experts
        expert_outputs = self._dispatch_and_combine(
            x_flat, routing_weights, expert_indices
        )
        
        # Handle attention mask - reconstruct full sequence
        if attention_mask is not None:
            output_flat = torch.zeros(
                batch_size * seq_len, self.hidden_dim,
                device=x.device, dtype=x.dtype
            )
            output_flat[valid_mask] = expert_outputs
            expert_outputs = output_flat
        
        # Reshape back: [batch, seq, hidden]
        output = expert_outputs.view(batch_size, seq_len, self.hidden_dim)
        
        # Output norm
        output = self.output_norm(output)
        
        # Residual
        output = residual + output
        
        # Routing stats
        routing_stats = {}
        if return_aux_loss:
            routing_stats = self.router.get_routing_stats(
                router_logits, expert_indices, self.top_k
            )
        
        return output, aux_loss, routing_stats
    
    def _dispatch_and_combine(
        self,
        x: torch.Tensor,           # [num_tokens, hidden_dim]
        routing_weights: torch.Tensor,  # [num_tokens, top_k]
        expert_indices: torch.Tensor,   # [num_tokens, top_k]
    ) -> torch.Tensor:
        """Dispatch tokens to experts and combine outputs."""
        num_tokens, hidden_dim = x.shape
        top_k = routing_weights.size(1)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            # expert_indices: [num_tokens, top_k]
            mask = (expert_indices == expert_idx)  # [num_tokens, top_k]
            token_indices, k_indices = mask.nonzero(as_tuple=True)
            
            if token_indices.numel() == 0:
                continue
            
            # Get tokens for this expert
            expert_tokens = x[token_indices]  # [num_expert_tokens, hidden_dim]
            
            # Get routing weights for these tokens
            weights = routing_weights[token_indices, k_indices].unsqueeze(-1)  # [num_expert_tokens, 1]
            
            # Expert forward pass
            expert_out = self.experts[expert_idx](expert_tokens)
            
            # Weighted output
            weighted_out = expert_out * weights
            
            # Accumulate
            output[token_indices] += weighted_out
        
        return output
    
    def _update_stats(self, expert_indices: torch.Tensor) -> None:
        """Update routing statistics."""
        self.total_tokens += expert_indices.size(0) * self.top_k
        for i in range(self.num_experts):
            count = (expert_indices == i).sum().item()
            self.expert_token_counts[i] += count
    
    def get_expert_utilization(self) -> torch.Tensor:
        """Get expert utilization percentages."""
        total = self.total_tokens.item()
        if total == 0:
            return torch.zeros(self.num_experts)
        return (self.expert_token_counts.float() / total * 100)
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.total_tokens.zero_()
        self.expert_token_counts.zero_()
    
    def get_parameter_count(self) -> dict:
        """Get parameter breakdown."""
        expert_params = sum(e.get_parameter_count() for e in self.experts)
        router_params = sum(p.numel() for p in self.router.parameters())
        norm_params = sum(p.numel() for p in self.input_norm.parameters())
        norm_params += sum(p.numel() for p in self.output_norm.parameters())
        
        return {
            "experts": expert_params,
            "router": router_params,
            "norms": norm_params,
            "total": expert_params + router_params + norm_params,
        }
    
    def load_expert_state_dict(self, expert_id: int, state_dict: dict) -> None:
        """Load state dict for a specific expert."""
        self.experts[expert_id].load_state_dict(state_dict)
    
    def get_expert_state_dict(self, expert_id: int) -> dict:
        """Get state dict for a specific expert."""
        return self.experts[expert_id].state_dict()


class SharedExpertMoELayer(MoELayer):
    """MoE layer with shared experts (always active).
    
    Some experts are designated as "shared" and process all tokens.
    This can improve training stability.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        num_shared_experts: int = 0,
        shared_experts: Optional[nn.ModuleList] = None,
        **kwargs,
    ):
        self.num_shared_experts = num_shared_experts
        # Adjust num_experts for router (router only routes to non-shared)
        super().__init__(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            expert_hidden_dim=expert_hidden_dim,
            shared_experts=shared_experts,
            **kwargs,
        )
        
        # Shared experts are the first num_shared_experts
        # They don't go through router
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        batch_size, seq_len, _ = x.shape
        residual = x
        
        # Input norm
        x = self.input_norm(x)
        x_flat = x.view(-1, self.hidden_dim)
        
        # Shared experts process all tokens
        shared_output = torch.zeros_like(x_flat)
        for i in range(self.num_shared_experts):
            shared_output += self.experts[i](x_flat)
        if self.num_shared_experts > 0:
            shared_output /= self.num_shared_experts
        
        # Router experts process subset
        if self.num_experts > self.num_shared_experts:
            # Filter valid tokens if mask provided
            if attention_mask is not None:
                valid_mask = attention_mask.view(-1).bool()
                router_input = x_flat[valid_mask]
            else:
                router_input = x_flat
                valid_mask = None
            
            if router_input.size(0) > 0:
                routing_weights, expert_indices, router_logits, aux_loss = self.router(
                    router_input, top_k=self.top_k, return_aux_loss=return_aux_loss
                )
                
                # Offset expert indices by num_shared_experts
                expert_indices = expert_indices + self.num_shared_experts
                
                # Dispatch
                router_output = self._dispatch_and_combine(
                    router_input, routing_weights, expert_indices
                )
                
                # Reconstruct full sequence
                if valid_mask is not None:
                    full_router_output = torch.zeros_like(x_flat)
                    full_router_output[valid_mask] = router_output
                    router_output = full_router_output
            else:
                router_output = torch.zeros_like(x_flat)
                aux_loss = None
                router_logits = None
                expert_indices = None
        else:
            router_output = torch.zeros_like(x_flat)
            aux_loss = None
            router_logits = None
            expert_indices = None
        
        # Combine shared + routed
        output_flat = shared_output + router_output
        
        # Output norm and residual
        output = output_flat.view(batch_size, seq_len, self.hidden_dim)
        output = self.output_norm(output)
        output = residual + output
        
        # Stats
        routing_stats = {}
        if return_aux_loss and router_logits is not None:
            routing_stats = self.router.get_routing_stats(
                router_logits, expert_indices - self.num_shared_experts, self.top_k
            )
        
        return output, aux_loss, routing_stats


def create_moe_layer(hidden_dim: int, config: dict, shared_experts: Optional[nn.ModuleList] = None) -> MoELayer:
    """Factory function to create MoE layer from config."""
    moe_type = config.get("moe_type", "standard")
    
    # Convert numeric config values
    expert_dropout = float(config.get("expert_dropout", 0.0))
    norm_eps = float(config.get("norm_eps", 1e-5))
    router_dropout = float(config.get("router_dropout", 0.0))
    capacity_factor = float(config.get("capacity_factor", 1.0))
    
    if moe_type == "shared_expert":
        return SharedExpertMoELayer(
            hidden_dim=hidden_dim,
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            expert_hidden_dim=config["expert_hidden_dim"],
            num_shared_experts=config.get("num_shared_experts", 0),
            expert_activation=config.get("expert_activation", "silu"),
            expert_norm=config.get("expert_norm", "rmsnorm"),
            expert_bias=config.get("expert_bias", False),
            expert_dropout=expert_dropout,
            norm_eps=norm_eps,
            router_hidden_dim=config.get("router_hidden_dim"),
            router_bias=config.get("router_bias", False),
            router_dropout=router_dropout,
            capacity_factor=capacity_factor,
            drop_tokens=config.get("drop_tokens", False),
            router_config=config.get("router_config"),
            shared_experts=shared_experts,
        )
    else:
        return MoELayer(
            hidden_dim=hidden_dim,
            num_experts=config["num_experts"],
            top_k=config["top_k"],
            expert_hidden_dim=config["expert_hidden_dim"],
            expert_activation=config.get("expert_activation", "silu"),
            expert_norm=config.get("expert_norm", "rmsnorm"),
            expert_bias=config.get("expert_bias", False),
            expert_dropout=expert_dropout,
            norm_eps=norm_eps,
            router_hidden_dim=config.get("router_hidden_dim"),
            router_bias=config.get("router_bias", False),
            router_dropout=router_dropout,
            capacity_factor=capacity_factor,
            drop_tokens=config.get("drop_tokens", False),
            router_config=config.get("router_config"),
            shared_experts=shared_experts,
        )