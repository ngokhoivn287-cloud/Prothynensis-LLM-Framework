"""Model Coordinator for MoMMs - orchestrates multiple models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model, ModelConfig, create_model
from .fusion import FusionLayer, FusionConfig, create_fusion
from .registry import ModelRegistry


@dataclass
class CoordinatorConfig:
    """Configuration for Model Coordinator."""
    hidden_dim: int = 768
    num_models: int = 32
    top_k: int = 2
    router_hidden_dim: Optional[int] = None
    router_type: str = "learned"  # "learned", "fixed", "adaptive"
    temperature: float = 1.0
    top_p: float = 1.0
    load_balancing: bool = True
    load_balancing_weight: float = 0.1
    confidence_threshold: float = 0.5
    use_confidence_gate: bool = True
    use_model_embeddings: bool = True
    model_embedding_dim: int = 64
    
    def to_dict(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_models": self.num_models,
            "top_k": self.top_k,
            "router_hidden_dim": self.router_hidden_dim,
            "router_type": self.router_type,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "load_balancing": self.load_balancing,
            "load_balancing_weight": self.load_balancing_weight,
            "confidence_threshold": self.confidence_threshold,
            "use_confidence_gate": self.use_confidence_gate,
            "use_model_embeddings": self.use_model_embeddings,
            "model_embedding_dim": self.model_embedding_dim,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "CoordinatorConfig":
        return cls(**d)


class ModelRouter(nn.Module):
    """
    Learned router for selecting top-K models per token.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_models: int,
        router_hidden_dim: Optional[int] = None,
        bias: bool = False,
        dropout: float = 0.0,
        use_model_embeddings: bool = True,
        model_embedding_dim: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_models = num_models
        self.router_hidden_dim = router_hidden_dim or hidden_dim
        self.use_model_embeddings = use_model_embeddings
        
        # Model embeddings for routing
        if use_model_embeddings:
            self.model_embeddings = nn.Embedding(num_models, model_embedding_dim)
            router_input_dim = hidden_dim + model_embedding_dim
        else:
            router_input_dim = hidden_dim
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, self.router_hidden_dim, bias=bias),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(self.router_hidden_dim, num_models, bias=bias),
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )
        
        # Initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch*seq, hidden_dim]
        top_k: int = 2,
        temperature: float = 1.0,
        top_p: float = 1.0,
        return_aux_loss: bool = True,
    ) -> Tuple[
        torch.Tensor,      # routing_weights: [batch*seq, top_k]
        torch.Tensor,      # model_indices: [batch*seq, top_k]
        torch.Tensor,      # router_logits: [batch*seq, num_models]
        Optional[torch.Tensor],  # aux_loss
        torch.Tensor,      # confidence: [batch*seq]
    ]:
        """
        Route tokens to top-K models.
        
        Returns:
            routing_weights: [batch*seq, top_k]
            model_indices: [batch*seq, top_k]
            router_logits: [batch*seq, num_models]
            aux_loss: Load balancing loss (or None)
            confidence: [batch*seq] - max router probability
        """
        batch_seq = x.size(0)
        
        # Router logits
        router_logits = self.router(x)  # [batch*seq, num_models]
        router_logits = router_logits / temperature
        
        # Softmax
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-K selection
        topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
        
        # Renormalize top-k
        routing_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        # Confidence (max probability)
        confidence = router_probs.max(dim=-1).values
        
        # Auxiliary load balancing loss
        aux_loss = None
        if self.training and return_aux_loss:
            aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        
        # Top-p filtering (optional)
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(router_probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            router_probs[indices_to_remove] = float("-inf")
            
            # Re-select top-k after top-p
            topk_probs, topk_indices = torch.topk(router_probs, k=top_k, dim=-1)
            routing_weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        
        return routing_weights, topk_indices, router_logits, aux_loss, confidence
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss."""
        num_tokens = router_probs.size(0)
        
        # Soft fraction
        expert_fraction = router_probs.mean(dim=0)
        
        # Hard fraction
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_(1, topk_indices, 1.0)
        expert_fraction_hard = expert_mask.mean(dim=0)
        
        # Load balancing loss
        aux_loss = self.num_models * (expert_fraction * expert_fraction_hard).sum()
        return aux_loss


class ModelCoordinator(nn.Module):
    """
    Model Coordinator for MoMMs.
    
    Orchestrates multiple models:
    1. Routes tokens to top-K models
    2. Fuses model outputs
    3. Computes load balancing
    4. Tracks routing statistics
    """
    
    def __init__(
        self,
        models: List[Model],
        config: CoordinatorConfig,
        fusion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.hidden_dim = config.hidden_dim
        self.top_k = config.top_k
        
        # Model embeddings for routing
        if config.use_model_embeddings:
            self.model_embeddings = nn.Embedding(
                config.num_models, config.model_embedding_dim
            )
        
        # Router
        self.router = ModelRouter(
            hidden_dim=config.hidden_dim,
            num_models=config.num_models,
            router_hidden_dim=config.router_hidden_dim,
            dropout=0.0,
            use_model_embeddings=config.use_model_embeddings,
            model_embedding_dim=config.model_embedding_dim,
        )
        
        # Fusion layer
        if fusion is not None:
            self.fusion = fusion
        else:
            fusion_config = FusionConfig(
                fusion_type="attention",
                hidden_dim=config.hidden_dim,
                num_models=config.num_models,
                top_k=config.top_k,
            )
            self.fusion = FusionLayer(FusionConfig.from_dict({
                "fusion_type": "attention",
                "hidden_dim": config.hidden_dim,
                "num_models": config.num_models,
                "top_k": config.top_k,
            }))
        
        # Model embeddings for routing (if not using separate embeddings)
        if not config.use_model_embeddings:
            self.register_buffer(
                "model_ids",
                torch.arange(config.num_models).long()
            )
        
        # Statistics tracking
        self.register_buffer("total_tokens", torch.zeros(1, dtype=torch.long))
        self.register_buffer("model_token_counts", torch.zeros(config.num_models, dtype=torch.long))
        self.register_buffer("model_confidence_sum", torch.zeros(config.num_models, dtype=torch.float))
        self.register_buffer("model_confidence_count", torch.zeros(config.num_models, dtype=torch.long))
        
        # Load balancing
        self.load_balancing = config.load_balancing
        self.load_balancing_weight = config.load_balancing_weight
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
        return_routing_stats: bool = True,
    ) -> Dict[str, Any]:
        """
        Coordinate models for a sequence.
        
        Args:
            hidden_states: [batch, seq, hidden]
            attention_mask: [batch, seq] (1 for valid, 0 for padding)
            return_aux_loss: Whether to compute load balancing loss
            return_routing_stats: Whether to return routing statistics
        
        Returns:
            Dict with:
                output: [batch, seq, hidden] - fused output
                aux_loss: Load balancing loss
                routing_weights: [batch, seq, top_k]
                model_indices: [batch, seq, top_k]
                routing_stats: Dict of routing statistics
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten for routing: [batch*seq, hidden]
        hidden_flat = hidden_states.view(-1, self.hidden_dim)
        
        # Handle attention mask
        if attention_mask is not None:
            valid_mask = attention_mask.view(-1).bool()
            hidden_flat = hidden_flat[valid_mask]
        
        num_tokens = hidden_flat.size(0)
        
        if num_tokens == 0:
            return {
                "output": hidden_states,
                "aux_loss": None,
                "routing_weights": None,
                "model_indices": None,
                "routing_stats": {},
            }
        
        # Route tokens
        routing_weights, model_indices, router_logits, aux_loss, confidence = self.router(
            hidden_flat,
            top_k=self.top_k,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            return_aux_loss=self.training,
        )
        
        # Update statistics
        if self.training:
            self._update_stats(model_indices)
        
        # Dispatch to models
        model_outputs = self._dispatch_and_combine(
            hidden_states, routing_weights, model_indices
        )
        
        # Create dense routing weights for fusion: [batch, seq, num_models]
        batch_size, seq_len, hidden_dim = hidden_states.shape
        dense_routing_weights = torch.zeros(batch_size, seq_len, self.num_models, device=hidden_states.device)
        
        if attention_mask is not None:
            valid_mask = attention_mask.view(-1).bool()
            valid_positions = valid_mask.nonzero(as_tuple=True)[0]
        else:
            valid_positions = torch.arange(batch_size * seq_len, device=hidden_states.device)
        
        for k in range(self.top_k):
            valid = model_indices[:, k] >= 0
            if valid.any():
                batch_seq_idx = valid_positions[valid]
                model_idx = model_indices[valid, k]
                weight = routing_weights[valid, k]
                dense_routing_weights[batch_seq_idx // seq_len, batch_seq_idx % seq_len, model_idx] = weight
        
        # Fuse outputs
        fused_output = self.fusion(
            model_outputs,
            routing_weights=dense_routing_weights,
        )
        
        # Prepare routing stats
        routing_stats = {}
        if return_routing_stats:
            routing_stats = self._compute_routing_stats(
                router_logits, model_indices, self.top_k
            )
        
        # Compute load balancing loss
        if self.training and self.load_balancing and aux_loss is not None:
            aux_loss = aux_loss * self.load_balancing_weight
        else:
            aux_loss = None
        
        return {
            "output": fused_output,
            "aux_loss": aux_loss,
            "routing_weights": routing_weights.view(-1, self.top_k),
            "model_indices": model_indices,
            "routing_stats": routing_stats,
        }
    
    def _dispatch_and_combine(
        self,
        hidden_states: torch.Tensor,  # [batch, seq, hidden]
        routing_weights: torch.Tensor,  # [batch*seq, top_k]
        model_indices: torch.Tensor,  # [batch*seq, top_k]
    ) -> List[torch.Tensor]:
        """Dispatch tokens to selected models and collect outputs."""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Flatten hidden states
        hidden_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_flat.size(0)
        
        # Initialize full-size outputs for all models
        model_outputs = [
            torch.zeros_like(hidden_flat)
            for _ in range(self.num_models)
        ]
        
        # Process each model
        for model_idx in range(self.num_models):
            # Find tokens routed to this model
            mask = (model_indices == model_idx)  # [num_tokens, top_k]
            token_indices, k_indices = mask.nonzero(as_tuple=True)
            
            if token_indices.numel() == 0:
                continue
            
            # Get tokens for this model
            model_tokens = hidden_flat[token_indices]  # [num_model_tokens, hidden]
            
            # Get routing weights for these tokens at their specific k position
            weights = routing_weights[token_indices, k_indices].unsqueeze(-1)  # [num_model_tokens, 1]
            
            # Run model on hidden states (skip embeddings, run transformer + final norm)
            model = self.models[model_idx]
            hidden_out = self._run_model_on_hidden_states(model, model_tokens)
            
            # Weighted output
            weighted_out = hidden_out * weights
            
            # Scatter back to full sequence
            full_out = torch.zeros_like(hidden_flat)
            full_out[token_indices] = weighted_out
            model_outputs[model_idx] = full_out.view(batch_size, seq_len, hidden_dim)
        
        return model_outputs
    
    def _run_model_on_hidden_states(
        self,
        model: Model,
        hidden_states: torch.Tensor,  # [num_tokens, hidden]
    ) -> torch.Tensor:
        """Run transformer layers on hidden states (skip embeddings)."""
        # hidden_states: [num_tokens, hidden]
        # Need [1, num_tokens, hidden] for transformer
        x = hidden_states.unsqueeze(0)
        
        # Run transformer layers
        for layer in model.layers:
            x, _ = layer(x, attention_mask=None, use_cache=False)
        
        # Final norm
        x = model.final_norm(x)
        
        return x.squeeze(0)  # [num_tokens, hidden]
    
    def _update_stats(self, model_indices: torch.Tensor) -> None:
        """Update routing statistics."""
        self.total_tokens += model_indices.size(0) * self.top_k
        for i in range(self.num_models):
            count = (model_indices == i).sum().item()
            self.model_token_counts[i] += count
    
    def _compute_routing_stats(
        self,
        router_logits: torch.Tensor,
        model_indices: torch.Tensor,
        top_k: int,
    ) -> dict:
        """Compute routing statistics for monitoring."""
        with torch.no_grad():
            router_probs = F.softmax(router_logits, dim=-1)
            num_tokens = router_probs.size(0)
            
            # Expert utilization
            expert_mask = torch.zeros_like(router_probs)
            expert_mask.scatter_(1, model_indices, 1.0)
            expert_counts = expert_mask.sum(dim=0)
            expert_utilization = expert_counts / (num_tokens * top_k) * 100
            
            # Routing entropy
            entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
            
            # Load balance
            mean_load = expert_counts.float().mean()
            std_load = expert_counts.float().std()
            cv = std_load / (mean_load + 1e-10)
            
            max_load = expert_counts.max().item()
            min_load = expert_counts.min().item()
            
            # Dead models
            dead_models = (expert_utilization < 1.0).sum().item()
            
            # Model collapse
            max_util = expert_utilization.max().item()
            model_collapse = max_util > 50.0
            
            # Confidence stats
            confidence = router_probs.max(dim=-1).values
            mean_confidence = confidence.mean().item()
            min_confidence = confidence.min().item()
            
            return {
                "model_utilization": expert_utilization.tolist(),
                "routing_entropy": entropy.item(),
                "load_balance_cv": cv.item(),
                "max_model_load": max_load,
                "min_model_load": min_load,
                "dead_models": dead_models,
                "model_collapse": model_collapse,
                "max_utilization": max_util,
                "mean_confidence": mean_confidence,
                "min_confidence": min_confidence,
            }
    
    def get_model_utilization(self) -> torch.Tensor:
        """Get model utilization percentages."""
        total = self.total_tokens.item()
        if total == 0:
            return torch.zeros(self.num_models)
        return (self.model_token_counts.float() / total * 100)
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self.total_tokens.zero_()
        self.model_token_counts.zero_()
        self.model_confidence_sum.zero_()
        self.model_confidence_count.zero_()
    
    def get_model(self, model_id: int) -> Model:
        """Get a specific model."""
        return self.models[model_id]
    
    def load_model_checkpoint(self, model_id: int, checkpoint_path: str) -> None:
        """Load checkpoint for a specific model."""
        model = self.models[model_id]
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    def save_model_checkpoint(self, model_id: int, checkpoint_path: str) -> None:
        """Save checkpoint for a specific model."""
        model = self.models[model_id]
        torch.save(model.state_dict(), checkpoint_path)
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using the coordinator.
        
        Uses the first model's embeddings and LM head for the token interface,
        with the coordinator routing and fusing hidden states.
        """
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        base_model = self.models[0]
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                # Embeddings from base model
                position_ids = torch.arange(
                    input_ids.size(1), dtype=torch.long, device=device
                ).unsqueeze(0).expand(batch_size, -1)
                token_emb = base_model.token_embedding(input_ids)
                pos_emb = base_model.position_embedding(position_ids)
                hidden_states = token_emb + pos_emb
                
                # Coordinator forward
                outputs = self.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    return_aux_loss=False,
                    return_routing_stats=False,
                )
                fused = outputs["output"]
                
                # LM head from base model
                if base_model.lm_head is not None:
                    logits = base_model.lm_head(fused)
                else:
                    logits = F.linear(fused, base_model.token_embedding.weight)
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(input_ids[i].tolist()):
                            next_token_logits[i, token_id] /= repetition_penalty
                
                # Top-k filtering
                if top_k is not None:
                    top_k_val = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1:]
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)
            
            # Check EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids

    @classmethod
    def from_config(
        cls,
        model_configs: List[ModelConfig],
        coordinator_config: CoordinatorConfig,
    ) -> "ModelCoordinator":
        """Create coordinator from model configs."""
        models = [create_model(c) for c in model_configs]
        return cls(models, coordinator_config)
    
    @classmethod
    def from_registry(
        cls,
        registry: ModelRegistry,
        coordinator_config: CoordinatorConfig,
    ) -> "ModelCoordinator":
        """Create coordinator from registry."""
        models = []
        for i in range(coordinator_config.num_models):
            model_id = f"model_{i:04d}"
            if model_id in registry.models:
                model = registry.load_model(model_id)
                models.append(model)
            else:
                raise ValueError(f"Model {model_id} not found in registry")
        return cls(models, coordinator_config)


def create_coordinator(
    models: List[Model],
    config: CoordinatorConfig | Dict[str, Any],
) -> ModelCoordinator:
    """Factory function to create coordinator."""
    if isinstance(config, dict):
        config = CoordinatorConfig(**config)
    return ModelCoordinator(models, config)