"""Expert pretraining trainer (Stage B)."""

from __future__ import annotations

import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any

from .base_trainer import BaseTrainer
from ..model import ExpertFFN, MoELanguageModel
from ..utils.logging import get_logger

logger = get_logger(__name__)


class ExpertTrainer(BaseTrainer):
    """Trainer for independent expert pretraining (Stage B).
    
    Trains a single expert as a standalone language model with its own
    embeddings and output head, using only its assigned data shard.
    The expert is trained WITHIN a minimal Transformer, but ONLY the 
    expert's FFN parameters receive gradients.
    """
    
    def __init__(
        self,
        expert: ExpertFFN,
        expert_id: int,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints/experts",
        log_dir: str = "logs/experts",
    ):
        # Wrap expert in a minimal language model for pretraining
        self.expert_id = expert_id
        self.expert = expert
        
        # Create a minimal model config for the expert
        model_config = config.get("model", {}) if config else {}
        vocab_size = model_config.get("vocab_size", 50304)
        max_seq_len = model_config.get("max_seq_len", 2048)
        hidden_dim = getattr(expert, 'ffn', expert).hidden_dim
        num_layers = model_config.get("expert_pretrain_num_layers", 4)  # Separate config
        num_heads = model_config.get("expert_pretrain_num_heads", 8)
        
        # Build a shallow transformer around the expert
        self.expert_model = self._build_expert_model(
            expert=expert,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        
        # CRITICAL: Freeze everything EXCEPT the expert FFN
        self._freeze_except_expert()
        
        super().__init__(
            model=self.expert_model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
            device=device,
            output_dir=f"{output_dir}/expert_{expert_id:03d}",
            log_dir=f"{log_dir}/expert_{expert_id:03d}",
            phase="expert_pretrain",
        )
        
        logger.info(f"Expert {expert_id} trainer initialized")
        logger.info(f"Expert parameters (trainable): {sum(p.numel() for p in expert.parameters() if p.requires_grad):,}")
        logger.info(f"Total model parameters: {sum(p.numel() for p in self.expert_model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.expert_model.parameters() if p.requires_grad):,}")
    
    def _freeze_except_expert(self) -> None:
        """Freeze all parameters except the expert FFN."""
        # First freeze everything
        for param in self.expert_model.parameters():
            param.requires_grad = False
        
        # Unfreeze ONLY the expert FFN parameters
        for param in self.expert.parameters():
            param.requires_grad = True
        
        # Also unfreeze embeddings and output head for language modeling
        self.expert_model.token_embedding.weight.requires_grad = True
        self.expert_model.position_embedding.weight.requires_grad = True
        self.expert_model.lm_head.weight.requires_grad = True
        
        trainable = sum(p.numel() for p in self.expert_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.expert_model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    
    def _build_expert_model(
        self,
        expert: ExpertFFN,
        vocab_size: int,
        max_seq_len: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> nn.Module:
        """Build a minimal transformer model around the expert for pretraining."""
        
        class ExpertPretrainModel(nn.Module):
            def __init__(self, expert, vocab_size, max_seq_len, hidden_dim, num_layers, num_heads):
                super().__init__()
                self.expert = expert
                self.hidden_dim = hidden_dim
                
                # Embeddings
                self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
                self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
                self.emb_dropout = nn.Dropout(0.1)
                
                # Transformer layers (shallow)
                self.layers = nn.ModuleList([
                    self._make_layer(hidden_dim, num_heads)
                    for _ in range(num_layers)
                ])
                
                # Final norm
                self.final_norm = nn.LayerNorm(hidden_dim)
                
                # LM head (tied)
                self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
                self.lm_head.weight = self.token_embedding.weight
            
            def _make_layer(self, hidden_dim, num_heads):
                from ..model.attention import CausalSelfAttention, RMSNorm
                return nn.ModuleDict({
                    "attn_norm": RMSNorm(hidden_dim),
                    "attention": CausalSelfAttention(hidden_dim, num_heads, hidden_dim // num_heads),
                    "ffn_norm": RMSNorm(hidden_dim),
                    "ffn": expert,  # Use the expert as FFN
                })
            
            def forward(self, input_ids, attention_mask=None, labels=None):
                batch_size, seq_len = input_ids.shape
                
                # Embeddings
                pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
                x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
                x = self.emb_dropout(x)
                
                # Transformer layers
                for layer in self.layers:
                    # Attention
                    residual = x
                    x = layer["attn_norm"](x)
                    attn_out, _ = layer["attention"](x, attention_mask)
                    x = residual + attn_out
                    
                    # FFN (expert)
                    residual = x
                    x = layer["ffn_norm"](x)
                    x = layer["ffn"](x)
                    x = residual + x
                
                # Final norm
                x = self.final_norm(x)
                
                # LM head
                logits = self.lm_head(x)
                
                # Compute loss
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                
                return {"logits": logits, "loss": loss, "total_loss": loss}
            
            def get_parameter_count(self):
                return {"total": sum(p.numel() for p in self.parameters())}
        
        return ExpertPretrainModel(expert, vocab_size, max_seq_len, hidden_dim, num_layers, num_heads)
    
    def setup_optimizer(self) -> None:
        """Setup optimizer for expert training - ONLY expert params + embeddings + head."""
        train_config = self.config.get("training", {}).get("expert_pretrain", {})
        
        lr = float(train_config.get("learning_rate", 3e-4))
        weight_decay = float(train_config.get("weight_decay", 0.1))
        beta1 = float(train_config.get("beta1", 0.9))
        beta2 = float(train_config.get("beta2", 0.95))
        
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
        
        # Cosine scheduler with warmup
        warmup_steps = int(train_config.get("warmup_steps", 2000))
        max_steps = int(train_config.get("max_steps", 100000))
        min_lr = float(train_config.get("min_lr", 3e-5))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(min_lr / lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for expert pretraining."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        
        outputs = self.model(input_ids=input_ids, labels=labels)
        return outputs
    
    def save_expert_only(self, step: int) -> str:
        """Save only the expert weights (for Stage D assembly)."""
        expert_path = Path(self.output_dir) / f"expert_{self.expert_id:03d}_step_{step:08d}.safetensors"
        expert_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save only expert FFN state dict (unwrap Expert wrapper)
        expert_state = self.expert.ffn.state_dict()
        import safetensors.torch
        safetensors.torch.save_file(expert_state, expert_path)
        
        logger.info(f"Saved expert {self.expert_id} weights to {expert_path}")
        return str(expert_path)
    
    @classmethod
    def from_config(
        cls,
        expert_id: int,
        expert: ExpertFFN,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> "ExpertTrainer":
        """Create trainer from config."""
        return cls(
            expert=expert,
            expert_id=expert_id,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
        )