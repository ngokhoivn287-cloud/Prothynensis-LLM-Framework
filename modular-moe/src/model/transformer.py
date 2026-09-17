"""Transformer block and stack."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .attention import CausalSelfAttention, RMSNorm
from .moe import MoELayer, create_moe_layer
from .expert import ExpertFFN


class TransformerBlock(nn.Module):
    """Single Transformer block with optional MoE."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        layer_idx: int,
        use_moe: bool = False,
        moe_config: Optional[dict] = None,
        shared_experts: Optional[nn.ModuleList] = None,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        use_flash_attn: bool = False,
        bias: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = use_moe
        
        # Attention norm
        self.attn_norm = RMSNorm(hidden_dim, norm_eps)
        
        # Attention
        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attn_dropout,
            bias=bias,
            use_flash_attn=use_flash_attn,
        )
        
        # Residual dropout
        self.resid_dropout = nn.Dropout(resid_dropout)
        
        # FFN norm
        self.ffn_norm = RMSNorm(hidden_dim, norm_eps)
        
        # FFN or MoE
        if use_moe and moe_config is not None:
            self.ffn = create_moe_layer(hidden_dim, moe_config, shared_experts=shared_experts)
            self.is_moe = True
        else:
            # Standard FFN (SwiGLU)
            expert_hidden_dim = moe_config.get("expert_hidden_dim", hidden_dim * 4) if moe_config else hidden_dim * 4
            self.ffn = nn.Sequential(
                RMSNorm(hidden_dim, norm_eps),
                nn.Linear(hidden_dim, expert_hidden_dim, bias=bias),
                nn.GELU(),
                nn.Linear(expert_hidden_dim, hidden_dim, bias=bias),
                nn.Dropout(resid_dropout),
            )
            self.is_moe = False
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_aux_loss: bool = True,
    ) -> Tuple[
        torch.Tensor,
        Optional[Tuple[torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
        dict,
    ]:
        """
        Returns:
            output: [batch, seq, hidden]
            present_key_value: Cached KV for inference
            aux_loss: MoE auxiliary loss (or None)
            routing_stats: Routing statistics (empty if not MoE)
        """
        # Attention block
        residual = x
        x = self.attn_norm(x)
        attn_output, present_key_value = self.attention(
            x, attention_mask=attention_mask,
            past_key_value=past_key_value, use_cache=use_cache
        )
        x = residual + self.resid_dropout(attn_output)
        
        # FFN/MoE block
        residual = x
        x = self.ffn_norm(x)
        
        if self.is_moe:
            ffn_output, aux_loss, routing_stats = self.ffn(
                x, attention_mask=attention_mask, return_aux_loss=return_aux_loss
            )
            x = residual + ffn_output
        else:
            x = residual + self.ffn(x)
            aux_loss = None
            routing_stats = {}
        
        return x, present_key_value, aux_loss, routing_stats


class TransformerStack(nn.Module):
    """Stack of Transformer blocks."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        moe_layer_indices: List[int],
        moe_config: dict,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        use_flash_attn: bool = False,
        bias: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        
        # Create shared experts for MoE layers (if any MoE layers exist)
        shared_experts = None
        if moe_layer_indices:
            # Create shared experts ONCE for all MoE layers
            expert_hidden_dim = moe_config["expert_hidden_dim"]
            num_experts = moe_config["num_experts"]
            expert_activation = moe_config.get("expert_activation", "silu")
            expert_norm = moe_config.get("expert_norm", "rmsnorm")
            expert_bias = moe_config.get("expert_bias", False)
            expert_dropout = moe_config.get("expert_dropout", 0.0)
            norm_eps = moe_config.get("norm_eps", 1e-5)
            
            shared_experts = nn.ModuleList([
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
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                layer_idx=i,
                use_moe=(i in moe_layer_indices),
                moe_config=moe_config if i in moe_layer_indices else None,
                shared_experts=shared_experts if i in moe_layer_indices else None,
                attn_dropout=attn_dropout,
                resid_dropout=resid_dropout,
                norm_type=norm_type,
                norm_eps=norm_eps,
                use_flash_attn=use_flash_attn,
                bias=bias,
            )
            for i in range(num_layers)
        ])
        
        # Final norm
        self.final_norm = RMSNorm(hidden_dim, norm_eps)
        self.shared_experts = shared_experts
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_aux_loss: bool = True,
    ) -> Tuple[
        torch.Tensor,
        Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
        Optional[torch.Tensor],
        List[dict],
    ]:
        """
        Returns:
            output: [batch, seq, hidden]
            present_key_values: List of cached KV for each layer
            total_aux_loss: Sum of all MoE auxiliary losses
            all_routing_stats: List of routing stats per layer
        """
        present_key_values = [] if use_cache else None
        all_routing_stats = []
        total_aux_loss = None
        
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            
            if self.gradient_checkpointing and self.training:
                # Gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    attention_mask,
                    past_kv,
                    use_cache,
                    return_aux_loss,
                    use_reentrant=False,
                )
                x, present_kv, aux_loss, routing_stats = outputs
            else:
                x, present_kv, aux_loss, routing_stats = layer(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    return_aux_loss=return_aux_loss,
                )
            
            if use_cache:
                present_key_values.append(present_kv)
            
            if aux_loss is not None:
                if total_aux_loss is None:
                    total_aux_loss = aux_loss
                else:
                    total_aux_loss = total_aux_loss + aux_loss
            
            all_routing_stats.append(routing_stats)
        
        # Final norm
        x = self.final_norm(x)
        
        return x, present_key_values, total_aux_loss, all_routing_stats
    
    def get_moe_layers(self) -> List[MoELayer]:
        """Get all MoE layers in the stack."""
        moe_layers = []
        for layer in self.layers:
            if layer.is_moe:
                moe_layers.append(layer.ffn)
        return moe_layers
    
    def get_shared_experts(self) -> Optional[nn.ModuleList]:
        """Get shared experts if they exist."""
        return self.shared_experts
    
    def get_parameter_count(self) -> dict:
        """Get parameter breakdown for the stack."""
        total = sum(p.numel() for p in self.parameters())
        
        # Count by component
        attn_params = 0
        moe_params = 0
        ffn_params = 0
        norm_params = 0
        
        # Track if we've counted shared experts already
        shared_experts_counted = False
        
        for layer in self.layers:
            # Attention
            attn_params += sum(p.numel() for p in layer.attention.parameters())
            attn_params += sum(p.numel() for p in layer.attn_norm.parameters())
            
            # FFN/MoE
            if layer.is_moe:
                if layer.ffn.owns_experts():
                    # This layer owns its experts (not shared)
                    moe_breakdown = layer.ffn.get_parameter_count()
                    moe_params += moe_breakdown["total"]
                else:
                    # Shared experts - only count once
                    if not shared_experts_counted:
                        moe_breakdown = layer.ffn.get_parameter_count()
                        moe_params += moe_breakdown["total"]
                        shared_experts_counted = True
                norm_params += sum(p.numel() for p in layer.ffn_norm.parameters())
            else:
                ffn_params += sum(p.numel() for p in layer.ffn.parameters())
                norm_params += sum(p.numel() for p in layer.ffn_norm.parameters())
        
        # Final norm
        norm_params += sum(p.numel() for p in self.final_norm.parameters())
        
        return {
            "total": total,
            "attention": attn_params,
            "moe": moe_params,
            "ffn": ffn_params,
            "norms": norm_params,
        }