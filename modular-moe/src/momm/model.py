"""Independent Model configuration and implementation for MoMMs."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for an independent MoMMs model."""
    model_id: str
    vocab_size: int = 50304
    max_seq_len: int = 2048
    hidden_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    head_dim: int = 64
    ffn_hidden_dim: int = 2048  # 4x hidden_dim
    activation: str = "silu"  # SwiGLU
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-5
    bias: bool = False
    dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    use_flash_attn: bool = False
    tie_embeddings: bool = True
    use_rmsnorm: bool = True
    gradient_checkpointing: bool = False
    
    # Specialization metadata
    specialization: str = "general"
    tokenizer_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "ffn_hidden_dim": self.ffn_hidden_dim,
            "activation": self.activation,
            "norm_type": self.norm_type,
            "norm_eps": self.norm_eps,
            "bias": self.bias,
            "dropout": self.dropout,
            "attn_dropout": self.attn_dropout,
            "resid_dropout": self.resid_dropout,
            "use_flash_attn": self.use_flash_attn,
            "tie_embeddings": self.tie_embeddings,
            "use_rmsnorm": self.use_rmsnorm,
            "gradient_checkpointing": self.gradient_checkpointing,
            "specialization": self.specialization,
            "tokenizer_id": self.tokenizer_id,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
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


class SwiGLU(nn.Module):
    """SwiGLU activation: x * SiLU(gate)"""
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * F.silu(gate)


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
        
        present_key_value = (k, v) if use_cache else None
        
        kv_seq_len = k.size(2)
        
        if self.use_flash_attn:
            causal_mask = self._get_causal_mask(kv_seq_len, x.device)
            causal_mask = causal_mask[:seq_len, :]
            
            if attention_mask is not None:
                attn_mask = attention_mask.view(x.size(0), 1, 1, seq_len).bool()
                if attn_mask.size(-1) < kv_seq_len:
                    pad = kv_seq_len - attn_mask.size(-1)
                    attn_mask = F.pad(attn_mask, (pad, 0), value=True)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) & attn_mask
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=causal_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            q = q * (self.head_dim ** -0.5)
            attn_scores = torch.matmul(q, k.transpose(-2, -1))
            
            causal_mask = self._get_causal_mask(kv_seq_len, x.device)
            causal_mask = causal_mask[:seq_len, :]
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))
            
            if attention_mask is not None:
                attn_mask = attention_mask.view(x.size(0), 1, 1, seq_len).bool()
                if attn_mask.size(-1) < kv_seq_len:
                    pad = kv_seq_len - attn_mask.size(-1)
                    attn_mask = F.pad(attn_mask, (pad, 0), value=True)
                attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
            attn_output = torch.matmul(attn_probs, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), seq_len, -1)
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output, present_key_value


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    
    def __init__(
        self,
        hidden_dim: int,
        ffn_hidden_dim: int,
        activation: str = "silu",
        norm_type: str = "rmsnorm",
        bias: bool = False,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout = dropout
        
        # Input normalization
        if norm_type.lower() == "rmsnorm":
            self.input_norm = RMSNorm(hidden_dim, norm_eps)
        elif norm_type.lower() == "layernorm":
            self.input_norm = nn.LayerNorm(hidden_dim, eps=norm_eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        # SwiGLU projections
        self.gate_proj = nn.Linear(hidden_dim, ffn_hidden_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, ffn_hidden_dim, bias=bias)
        
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
        self.down_proj = nn.Linear(ffn_hidden_dim, hidden_dim, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Output normalization
        self.output_norm = RMSNorm(hidden_dim, 1e-5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Input norm
        x = self.input_norm(x)
        
        # Gate and up projections
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        
        # SwiGLU
        x = up * F.silu(gate)
        
        # Dropout
        x = F.dropout(x, p=0.0, training=self.training)
        
        # Down projection
        x = self.down_proj(x)
        
        # Output norm
        x = self.output_norm(x)
        
        # Residual
        return residual + x


class TransformerBlock(nn.Module):
    """Single Transformer block with attention + FFN."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        layer_idx: int,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        norm_type: str = "rmsnorm",
        norm_eps: float = 1e-5,
        use_flash_attn: bool = False,
        bias: bool = False,
        activation: str = "silu",
        norm_type_ffn: str = "rmsnorm",
        ffn_norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        
        # Attention
        self.attn_norm = RMSNorm(hidden_dim, 1e-5)
        self.attention = CausalSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=attn_dropout,
            bias=False,
            use_flash_attn=use_flash_attn,
        )
        
        # Residual dropout
        self.resid_dropout = nn.Dropout(resid_dropout)
        
        # FFN
        self.ffn_norm = RMSNorm(hidden_dim, 1e-5)
        self.ffn = SwiGLUFFN(
            hidden_dim=hidden_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            activation="silu",
            norm_type="rmsnorm",
            bias=False,
            dropout=0.0,
            norm_eps=1e-5,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Attention block
        residual = x
        x = self.attn_norm(x)
        attn_output, present_key_value = self.attention(
            x, attention_mask=attention_mask,
            past_key_value=past_key_value, use_cache=use_cache
        )
        x = residual + F.dropout(attn_output, p=0.0, training=self.training)
        
        # FFN block
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x, present_key_value


class Model(nn.Module):
    """
    Independent MoMMs Model.
    
    Each model is a complete, independently trainable decoder-only Transformer
    with its own embeddings, attention layers, FFN, and LM head.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self._model_id = config.model_id
        
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_seq_len
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.ffn_hidden_dim = config.ffn_hidden_dim
        self.tie_embeddings = config.tie_embeddings
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_dim)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(config.resid_dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                head_dim=config.head_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                layer_idx=i,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                resid_dropout=config.resid_dropout,
                norm_type=config.norm_type,
                norm_eps=config.norm_eps,
                use_flash_attn=config.use_flash_attn,
                bias=config.bias,
                activation=config.activation,
            )
            for i in range(config.num_layers)
        ])
        
        # Final norm
        self.final_norm = nn.LayerNorm(config.hidden_dim, eps=config.norm_eps)
        
        # LM head
        if config.tie_embeddings:
            self.lm_head = None  # Will use token_embedding.weight
        else:
            self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Store config for checkpointing
        self._config_dict = config.to_dict()
    
    def _init_weights(self) -> None:
        # Embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Initialize transformer blocks
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
    
    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        return self._config_dict.copy()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        
        # Position ids
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = F.dropout(token_emb + pos_emb, p=self.config.resid_dropout, training=self.training)
        
        # Transformer layers
        present_key_values = [] if use_cache else None
        past_kvs = past_key_values if past_key_values else [None] * self.config.num_layers
        
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_key_values else None
            
            if self.gradient_checkpointing and self.training:
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
                    use_reentrant=False,
                )
                x, present_kv = outputs
            else:
                x, present_kv = self.layers[i](
                    x,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )
            
            if use_cache:
                present_key_values.append(present_kv)
        
        # Final norm
        x = self.final_norm(x)
        
        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = F.linear(x, self.token_embedding.weight)
        
        output = {
            "logits": logits,
            "present_key_values": present_key_values,
        }
        
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="mean",
            )
            
            output["loss"] = loss
            output["total_loss"] = loss
        
        return output
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        past_key_values = None
        
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.forward(
                    input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            
            logits = outputs["logits"]
            past_key_values = outputs["present_key_values"]
            
            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(input_ids[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k is not None:
                top_k = min(top_k, next_token_logits.size(-1))
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1:]
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Top-p (nucleus) filtering
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
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return input_ids
    
    def get_parameter_count(self) -> Dict[str, int]:
        breakdown = {}
        
        # Embeddings
        embed_params = self.token_embedding.weight.numel()
        pos_params = self.position_embedding.weight.numel()
        breakdown["token_embedding"] = embed_params
        breakdown["position_embedding"] = pos_params
        breakdown["embeddings_total"] = embed_params + pos_params
        
        # Transformer layers - compute per layer without double-counting
        attn_params = 0
        ffn_params = 0
        norm_params = 0
        
        for layer in self.layers:
            # Attention block: q_proj, k_proj, v_proj, out_proj, attn_norm
            attn_params += sum(p.numel() for p in layer.attention.parameters())
            attn_params += sum(p.numel() for p in layer.attn_norm.parameters())
            
            # FFN block: input_norm, gate_proj, up_proj, down_proj, output_norm
            ffn_params += sum(p.numel() for p in layer.ffn.parameters())
            
            # Additional norms: ffn_norm only (attn_norm already counted in attn_params)
            norm_params += sum(p.numel() for p in layer.ffn_norm.parameters())
        
        # Final norm
        norm_params += sum(p.numel() for p in self.final_norm.parameters())
        
        breakdown["attention"] = attn_params
        breakdown["ffn"] = ffn_params
        breakdown["norms"] = norm_params
        
        # LM head
        if self.lm_head is not None:
            lm_head_params = self.lm_head.weight.numel()
        else:
            lm_head_params = 0  # Tied
        breakdown["lm_head"] = lm_head_params
        
        # Total - use actual parameter count to avoid double-counting
        breakdown["total"] = sum(p.numel() for p in self.parameters())
        
        return breakdown
    
    def print_parameter_count(self) -> None:
        breakdown = self.get_parameter_count()
        
        print("\n" + "=" * 60)
        print(f"MODEL {self.model_id} PARAMETER COUNT")
        print("=" * 60)
        
        for key, value in breakdown.items():
            if key == "total":
                continue
            print(f"{key:30s}: {value:>15,} ({value/1e6:>8.2f}M)")
        
        print("-" * 60)
        print(f"{'TOTAL':30s}: {breakdown['total']:>15,} ({breakdown['total']/1e9:>8.2f}B)")
        print("=" * 60)
    
    @property
    def model_id(self) -> str:
        return self._model_id


def create_model(config: ModelConfig | Dict[str, Any]) -> Model:
    """Factory function to create a model from config."""
    if isinstance(config, dict):
        config = ModelConfig(**config)
    return Model(config)