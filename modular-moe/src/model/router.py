"""Router for Mixture-of-Experts."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Router(nn.Module):
    """Learned router for Top-K expert selection."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        router_hidden_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.router_hidden_dim = router_hidden_dim or hidden_dim
        
        # Router network: hidden_dim -> router_hidden_dim -> num_experts
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, self.router_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.router_hidden_dim, num_experts, bias=bias),
        )
        
        # Initialize router weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize router weights for stable routing."""
        for module in self.router.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        top_k: int = 2,
        return_aux_loss: bool = True,
    ) -> Tuple[
        torch.Tensor,      # routing_weights: [batch*seq, top_k]
        torch.Tensor,      # expert_indices: [batch*seq, top_k]
        torch.Tensor,      # router_logits: [batch*seq, num_experts]
        Optional[torch.Tensor],  # aux_loss
    ]:
        """
        Args:
            x: [batch_size * seq_len, hidden_dim] - flattened token representations
            top_k: Number of experts to route to
            return_aux_loss: Whether to compute load balancing loss
        
        Returns:
            routing_weights: [batch*seq, top_k] - softmax probabilities for selected experts
            expert_indices: [batch*seq, top_k] - indices of selected experts
            router_logits: [batch*seq, num_experts] - raw logits
            aux_loss: Load balancing auxiliary loss (or None)
        """
        # [batch*seq, num_experts]
        router_logits = self.router(x)
        
        # Top-K selection
        # router_probs: [batch*seq, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K values and indices
        topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
        
        # Renormalize top-k probabilities
        routing_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Auxiliary load balancing loss
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        
        return routing_weights, topk_indices, router_logits, aux_loss
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.
        
        Loss encourages uniform expert utilization:
        L_aux = num_experts * sum(mean(router_probs) * mean(expert_mask))
        """
        # router_probs: [num_tokens, num_experts]
        # topk_indices: [num_tokens, top_k]
        
        num_tokens = router_probs.size(0)
        
        # Fraction of tokens routed to each expert (soft)
        expert_fraction = router_probs.mean(dim=0)  # [num_experts]
        
        # Fraction of tokens routed to each expert (hard, via top-k)
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(1, topk_indices, 1.0)
        expert_fraction_hard = expert_mask.mean(dim=0)  # [num_experts]
        
        # Load balancing loss
        aux_loss = self.num_experts * (expert_fraction * expert_fraction_hard).sum()
        
        return aux_loss
    
    def get_routing_stats(
        self,
        router_logits: torch.Tensor,
        topk_indices: torch.Tensor,
        top_k: int,
    ) -> dict:
        """Compute routing statistics for monitoring."""
        with torch.no_grad():
            router_probs = F.softmax(router_logits, dim=-1)
            num_tokens = router_probs.size(0)
            
            # Expert utilization
            expert_mask = torch.zeros_like(router_probs)
            expert_mask.scatter_(1, topk_indices, 1.0)
            expert_counts = expert_mask.sum(dim=0)  # [num_experts]
            expert_utilization = expert_counts / (num_tokens * top_k) * 100
            
            # Routing entropy
            entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
            
            # Load balance metrics
            mean_load = expert_counts.float().mean()
            std_load = expert_counts.float().std()
            cv = std_load / (mean_load + 1e-10)
            
            max_load = expert_counts.max().item()
            min_load = expert_counts.min().item()
            
            # Dead experts (utilization < 1%)
            dead_experts = (expert_utilization < 1.0).sum().item()
            
            # Expert collapse (one expert gets > 50%)
            max_util = expert_utilization.max().item()
            expert_collapse = max_util > 50.0
            
            return {
                "expert_utilization": expert_utilization.tolist(),
                "routing_entropy": entropy.item(),
                "load_balance_cv": cv.item(),
                "max_expert_load": max_load,
                "min_expert_load": min_load,
                "dead_experts": dead_experts,
                "expert_collapse": expert_collapse,
                "max_utilization": max_util,
            }


class NoisyRouter(Router):
    """Router with noise injection for exploration during training."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        router_hidden_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        noise_std: float = 1.0,
    ):
        super().__init__(hidden_dim, num_experts, router_hidden_dim, bias, dropout)
        self.noise_std = noise_std
        self.noise_layer = nn.Linear(self.router_hidden_dim, num_experts, bias=bias)
        nn.init.zeros_(self.noise_layer.weight)
        if self.noise_layer.bias is not None:
            nn.init.zeros_(self.noise_layer.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        top_k: int = 2,
        return_aux_loss: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        # Clean logits
        clean_logits = self.router(x)
        
        # Noisy logits (only during training)
        if self.training and self.noise_std > 0:
            noise_logits = self.noise_layer(x)
            noise = torch.randn_like(noise_logits) * self.noise_std
            router_logits = clean_logits + noise * F.softplus(noise_logits)
        else:
            router_logits = clean_logits
        
        # Top-K selection
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
        routing_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        
        return routing_weights, topk_indices, router_logits, aux_loss


def create_router(hidden_dim: int, num_experts: int, config: dict) -> Router:
    """Factory function to create router from config."""
    router_type = config.get("router_type", "standard")
    
    if router_type == "noisy":
        return NoisyRouter(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            router_hidden_dim=config.get("router_hidden_dim"),
            bias=config.get("router_bias", False),
            dropout=config.get("router_dropout", 0.0),
            noise_std=config.get("router_noise_std", 1.0),
        )
    else:
        return Router(
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            router_hidden_dim=config.get("router_hidden_dim"),
            bias=config.get("router_bias", False),
            dropout=config.get("router_dropout", 0.0),
        )