"""Complete MoE Language Model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from .transformer import TransformerStack
from .moe import MoELayer
from .attention import RMSNorm
from src.utils.config import convert_config_types


class MoELanguageModel(nn.Module):
    """Decoder-only MoE language model."""
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        num_experts: int,
        top_k: int,
        expert_hidden_dim: int,
        expert_activation: str = "silu",
        expert_norm: str = "rmsnorm",
        expert_bias: bool = False,
        expert_dropout: float = 0.0,
        router_hidden_dim: Optional[int] = None,
        router_bias: bool = False,
        router_dropout: float = 0.0,
        tie_embeddings: bool = True,
        use_rmsnorm: bool = True,
        norm_eps: float = 1e-5,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
        causal: bool = True,
        use_flash_attn: bool = False,
        gradient_checkpointing: bool = False,
        moe_layer_indices: Optional[List[int]] = None,
        moe_config: Optional[dict] = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.tie_embeddings = tie_embeddings
        self.gradient_checkpointing = gradient_checkpointing
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Position embeddings (learned)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(resid_dropout)
        
        # Default MoE layer indices: every layer if not specified
        if moe_layer_indices is None:
            moe_layer_indices = list(range(num_layers))
        
        # Default MoE config
        if moe_config is None:
            moe_config = {
                "num_experts": num_experts,
                "top_k": top_k,
                "expert_hidden_dim": expert_hidden_dim,
                "expert_activation": expert_activation,
                "expert_norm": expert_norm,
                "expert_bias": expert_bias,
                "expert_dropout": expert_dropout,
                "norm_eps": norm_eps,
                "router_hidden_dim": router_hidden_dim,
                "router_bias": router_bias,
                "router_dropout": router_dropout,
            }
        
        # Transformer stack
        self.transformer = TransformerStack(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            moe_layer_indices=moe_layer_indices,
            moe_config=moe_config,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
            norm_type="rmsnorm" if use_rmsnorm else "layernorm",
            norm_eps=norm_eps,
            use_flash_attn=use_flash_attn,
            bias=False,
            gradient_checkpointing=gradient_checkpointing,
        )
        
        # LM head
        if tie_embeddings:
            self.lm_head = None  # Will use token_embedding.weight
        else:
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Initialize weights
        self._init_weights()
        
        # Store config for checkpointing
        self.config = {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "expert_hidden_dim": expert_hidden_dim,
            "expert_activation": expert_activation,
            "expert_norm": expert_norm,
            "expert_bias": expert_bias,
            "expert_dropout": expert_dropout,
            "router_hidden_dim": router_hidden_dim,
            "router_bias": router_bias,
            "router_dropout": router_dropout,
            "tie_embeddings": tie_embeddings,
            "use_rmsnorm": use_rmsnorm,
            "norm_eps": norm_eps,
            "attn_dropout": attn_dropout,
            "resid_dropout": resid_dropout,
            "causal": causal,
            "use_flash_attn": use_flash_attn,
            "gradient_checkpointing": gradient_checkpointing,
            "moe_layer_indices": moe_layer_indices,
        }
    
    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        if self.lm_head is not None:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        
        # Initialize transformer blocks
        for module in self.transformer.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
    
    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding
    
    def set_input_embeddings(self, embeddings: nn.Embedding) -> None:
        self.token_embedding = embeddings
    
    def get_output_embeddings(self) -> Optional[nn.Linear]:
        return self.lm_head
    
    def tie_weights(self) -> None:
        """Tie input and output embeddings."""
        if self.lm_head is not None:
            self.lm_head.weight = self.token_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        return_aux_loss: bool = True,
        labels: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (1 for valid, 0 for padding)
            past_key_values: List of cached KV for each layer
            use_cache: Whether to return KV cache
            return_aux_loss: Whether to compute MoE auxiliary loss
            labels: [batch_size, seq_len] for computing loss
            position_ids: Optional [batch_size, seq_len] for position embeddings
        
        Returns:
            Dict with:
                logits: [batch, seq, vocab]
                loss: Cross-entropy loss (if labels provided)
                aux_loss: MoE auxiliary loss
                total_loss: loss + aux_loss
                present_key_values: KV cache
                routing_stats: List of routing stats per layer
        """
        batch_size, seq_len = input_ids.shape
        
        # Position ids
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        x = self.emb_dropout(token_emb + pos_emb)
        
        # Transformer
        x, present_key_values, aux_loss, routing_stats = self.transformer(
            x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_aux_loss=return_aux_loss,
        )
        
        # LM head
        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            # Tied embeddings
            logits = F.linear(x, self.token_embedding.weight)
        
        output = {
            "logits": logits,
            "aux_loss": aux_loss,
            "present_key_values": present_key_values,
            "routing_stats": routing_stats,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Cross entropy loss
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction="mean",
            )
            
            output["loss"] = loss
            
            # Total loss with auxiliary
            if aux_loss is not None:
                output["total_loss"] = loss + aux_loss
            else:
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
        
        # Prepare attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Only pass the last token when using past_key_values to avoid duplicate KV
            if past_key_values is not None:
                step_input_ids = input_ids[:, -1:]
                step_attention_mask = attention_mask[:, -1:] if attention_mask.size(1) > 1 else attention_mask
                past_len = past_key_values[0][0].size(2)
                step_position_ids = torch.arange(past_len, past_len + 1, dtype=torch.long, device=device).unsqueeze(0)
            else:
                step_input_ids = input_ids
                step_attention_mask = attention_mask
                step_position_ids = None
            
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    step_input_ids,
                    attention_mask=step_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_aux_loss=False,
                    position_ids=step_position_ids,
                )
            
            logits = outputs["logits"]
            past_key_values = outputs["present_key_values"]
            
            # Get last token logits
            next_token_logits = logits[:, -1, :]
            if temperature != 0.0:
                next_token_logits = next_token_logits / temperature
            
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
            
            # Sample or argmax
            if temperature == 0.0:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e4, posinf=-1e4, neginf=-1e4)
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
            ], dim=-1)
            
            # Check EOS (stop if ANY sequence in the batch generated EOS)
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break
        
        return input_ids
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get detailed parameter count breakdown."""
        breakdown = {}
        
        # Embeddings
        embed_params = self.token_embedding.weight.numel()
        pos_params = self.position_embedding.weight.numel()
        breakdown["token_embedding"] = embed_params
        breakdown["position_embedding"] = pos_params
        breakdown["embeddings_total"] = embed_params + pos_params
        
        # Transformer stack
        stack_breakdown = self.transformer.get_parameter_count()
        
        # Separate expert params from MoE total
        # MoE total = experts + router + norms (input/output norms in MoE layers)
        # We need to extract just expert params
        shared_experts = self.transformer.get_shared_experts()
        if shared_experts is not None:
            shared_expert_params = sum(p.numel() for p in shared_experts.parameters())
            breakdown["shared_experts"] = shared_expert_params
            # Router params = MoE total - expert params - MoE norms
            # But we don't easily have MoE norms separate. Let's use stack breakdown.
            breakdown["moe_total"] = stack_breakdown.get("moe", 0)
            breakdown["router_and_moe_norms"] = breakdown["moe_total"] - shared_expert_params
        else:
            breakdown["moe_total"] = stack_breakdown.get("moe", 0)
        
        breakdown["attention"] = stack_breakdown.get("attention", 0)
        breakdown["ffn"] = stack_breakdown.get("ffn", 0)
        breakdown["norms"] = stack_breakdown.get("norms", 0)
        
        # LM head
        if self.lm_head is not None:
            lm_head_params = self.lm_head.weight.numel()
        else:
            lm_head_params = 0  # Tied
        breakdown["lm_head"] = lm_head_params
        
        # Total
        breakdown["total"] = sum(v for k, v in breakdown.items() if k != "total")
        
        return breakdown
    
    def print_parameter_count(self) -> None:
        """Print parameter count breakdown."""
        breakdown = self.get_parameter_count()
        
        print("\n" + "=" * 60)
        print("PARAMETER COUNT BREAKDOWN")
        print("=" * 60)
        
        # Print in required format
        # SHARED PARAMETERS
        shared_params = breakdown.get("embeddings_total", 0) + breakdown.get("attention", 0) + breakdown.get("norms", 0) + breakdown.get("ffn", 0)
        if "shared_experts" in breakdown:
            shared_params += breakdown["shared_experts"]
        
        expert_params = breakdown.get("shared_experts", 0)
        if expert_params == 0 and "moe_total" in breakdown:
            expert_params = breakdown["moe_total"] - breakdown.get("router_and_moe_norms", 0)
        
        router_params = breakdown.get("router_and_moe_norms", 0)
        
        print(f"{'SHARED PARAMETERS':30s}: {shared_params:>15,} ({shared_params/1e6:>8.2f}M)")
        print(f"{'EXPERT PARAMETERS':30s}: {expert_params:>15,} ({expert_params/1e6:>8.2f}M)")
        print(f"{'ROUTER PARAMETERS':30s}: {router_params:>15,} ({router_params/1e6:>8.2f}M)")
        print(f"{'LM HEAD':30s}: {breakdown.get('lm_head', 0):>15,} ({breakdown.get('lm_head', 0)/1e6:>8.2f}M)")
        print("-" * 60)
        print(f"{'TOTAL MoE PARAMETERS':30s}: {breakdown.get('moe_total', 0):>15,} ({breakdown.get('moe_total', 0)/1e9:>8.2f}B)")
        print(f"{'TOTAL MODEL':30s}: {breakdown['total']:>15,} ({breakdown['total']/1e9:>8.2f}B)")
        print("=" * 60)
        
        # Also print detailed breakdown
        print("\nDETAILED BREAKDOWN:")
        for key, value in breakdown.items():
            if key == "total":
                continue
            print(f"{key:30s}: {value:>15,} ({value/1e6:>8.2f}M)")
    
    def get_moe_layers(self) -> List[MoELayer]:
        """Get all MoE layers."""
        return self.transformer.get_moe_layers()
    
    def load_expert_checkpoint(self, expert_id: int, checkpoint_path: str) -> None:
        """Load a single expert from checkpoint."""
        moe_layers = self.get_moe_layers()
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            import safetensors.torch
            state_dict = safetensors.torch.load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, weights_only=False)
        for moe_layer in moe_layers:
            moe_layer.load_expert_state_dict(expert_id, state_dict)
    
    def save_expert_checkpoint(self, expert_id: int, checkpoint_path: str) -> None:
        """Save a single expert to checkpoint."""
        moe_layers = self.get_moe_layers()
        for moe_layer in moe_layers:
            state_dict = moe_layer.get_expert_state_dict(expert_id)
            torch.save(state_dict, checkpoint_path)
    
    @classmethod
    def from_config(cls, config: dict) -> "MoELanguageModel":
        """Create model from config dict."""
        model_config = config.get("model", config)
        model_config = convert_config_types(model_config)
        return cls(**model_config)
    
    def get_config(self) -> dict:
        """Get model configuration."""
        return self.config.copy()


def _extract_model_config(config: dict) -> dict:
    """Extract model constructor kwargs from a system or model config dict."""
    model_config = config.get("model", None)
    if model_config is None:
        model_config = config.get("solver", config)
    model_config = convert_config_types(model_config)
    if not isinstance(model_config, dict):
        return {}

    result = dict(model_config)

    # Map solver-style keys to MoELanguageModel constructor names
    key_map = {
        "ffn_hidden_dim": "expert_hidden_dim",
        "activation": "expert_activation",
        "norm_type": "expert_norm",
        "dropout": "expert_dropout",
    }
    for src, dst in key_map.items():
        if src in result and dst not in result:
            result[dst] = result.pop(src)

    # Ensure required MoE fields have defaults when missing
    result.setdefault("num_experts", 1)
    result.setdefault("top_k", 1)
    result.setdefault("expert_hidden_dim", result.get("hidden_dim", 512) * 4)
    result.setdefault("expert_activation", "silu")
    result.setdefault("expert_norm", "rmsnorm")
    result.setdefault("expert_bias", False)
    result.setdefault("expert_dropout", 0.0)
    result.setdefault("router_hidden_dim", None)
    result.setdefault("router_bias", False)
    result.setdefault("router_dropout", 0.0)
    result.setdefault("tie_embeddings", True)
    result.setdefault("use_rmsnorm", True)
    result.setdefault("norm_eps", 1e-5)
    result.setdefault("attn_dropout", 0.0)
    result.setdefault("resid_dropout", 0.0)
    result.setdefault("causal", True)
    result.setdefault("use_flash_attn", False)
    result.setdefault("gradient_checkpointing", False)
    result.setdefault("moe_layer_indices", None)
    result.setdefault("moe_config", None)

    # Drop non-model keys that may be present in system configs
    for key in [
        "count", "parameter_count_per_model", "total_parameters",
        "architecture", "orchestration", "system", "datasets",
        "training", "recruitment", "cache", "inference",
    ]:
        result.pop(key, None)

    return result


def create_model_from_config(config: dict) -> MoELanguageModel:
    """Factory function to create model from full config."""
    model_config = _extract_model_config(config)
    return MoELanguageModel(**model_config)