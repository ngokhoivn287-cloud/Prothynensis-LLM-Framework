"""Multi-head attention with causal masking."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn and hasattr(F, "scaled_dot_product_attention")
        
        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask buffer
        self.register_buffer("causal_mask", None, persistent=False)
    
    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or create causal mask."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: [batch_size, seq_len, hidden_dim]
            attention_mask: Optional [batch_size, seq_len] (1 for valid, 0 for padding)
            past_key_value: Optional cached (key, value) for inference
            use_cache: Whether to return cached key/values
        
        Returns:
            output: [batch_size, seq_len, hidden_dim]
            present_key_value: Cached (key, value) if use_cache
        """
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key/values for inference
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
            past_len = past_k.size(2)
        else:
            past_len = 0
        
        present_key_value = (k, v) if use_cache else None
        
        # Actual sequence length (including past)
        kv_seq_len = k.size(2)
        
        if self.use_flash_attn:
            # Use PyTorch's scaled_dot_product_attention (Flash Attention v2)
            # Create causal mask for query x key
            causal_mask = self._get_causal_mask(kv_seq_len, x.device)
            # For new tokens, they should attend to all past tokens + previous new tokens
            causal_mask = causal_mask[past_len:past_len + seq_len, :]
            
            # Combine with attention mask if provided
            if attention_mask is not None:
                # Expand attention mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
                attn_mask = attention_mask.view(batch_size, 1, 1, seq_len).bool()
                # Pad to kv_seq_len if needed
                if attn_mask.size(-1) < kv_seq_len:
                    pad = kv_seq_len - attn_mask.size(-1)
                    attn_mask = F.pad(attn_mask, (pad, 0), value=True)
                # Combine with causal mask
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) & attn_mask
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention computation
            # Scale
            q = q * (self.head_dim ** -0.5)
            
            # Attention scores: [batch, heads, seq_len, kv_seq_len]
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            
            # Causal mask - new tokens attend to all past + previous new tokens
            causal_mask = self._get_causal_mask(kv_seq_len, x.device)
            causal_mask = causal_mask[past_len:past_len + seq_len, :]  # [seq_len, kv_seq_len]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, kv_seq_len]
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
            
            # Attention mask (padding)
            if attention_mask is not None:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attn_mask = attention_mask.view(batch_size, 1, 1, seq_len).bool()
                if attn_mask.size(-1) < kv_seq_len:
                    pad = kv_seq_len - attn_mask.size(-1)
                    attn_mask = F.pad(attn_mask, (pad, 0), value=True)
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
            
            # Softmax
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.attn_dropout(attn_probs)
            
            # Apply to values
            attn_output = torch.matmul(attn_probs, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, present_key_value


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def get_norm_layer(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
    """Get normalization layer by name."""
    if norm_type.lower() == "rmsnorm":
        return RMSNorm(dim, eps)
    elif norm_type.lower() == "layernorm":
        return nn.LayerNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm type: {norm_type}")