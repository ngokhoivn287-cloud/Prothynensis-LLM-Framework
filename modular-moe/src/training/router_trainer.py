"""Router training trainer (Stage E)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List

from .base_trainer import BaseTrainer
from ..model import MoELanguageModel, MoELayer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class RouterTrainer(BaseTrainer):
    """Trainer for router training (Stage E).
    
    Freezes experts and shared components, trains only the router
    to learn which experts should process each token.
    """
    
    def __init__(
        self,
        model: MoELanguageModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints/router",
        log_dir: str = "logs/router",
    ):
        self.moe_model = model
        
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
            device=device,
            output_dir=output_dir,
            log_dir=log_dir,
            phase="router_train",
        )
        
        # Freeze experts and shared components after BaseTrainer sets self.config/model
        self._freeze_experts_and_shared()
        
        logger.info("Router trainer initialized")
        self._log_trainable_params()
    
    def _freeze_experts_and_shared(self) -> None:
        """Freeze all parameters except router."""
        train_config = self.config.get("training", {}).get("router_train", {})
        freeze_experts = train_config.get("freeze_experts", True)
        freeze_shared = train_config.get("freeze_shared", True)
        
        # Count frozen vs trainable
        frozen = 0
        trainable = 0
        
        for name, param in self.model.named_parameters():
            should_freeze = False
            
            if freeze_experts and ("expert" in name.lower() or "ffn" in name.lower()):
                should_freeze = True
            elif freeze_shared and ("router" not in name.lower()):
                # Freeze everything except router
                # But keep LM head and embeddings trainable? 
                # Usually we freeze those too for pure router training
                if "embed" not in name.lower() and "lm_head" not in name.lower():
                    should_freeze = True
            
            if should_freeze:
                param.requires_grad = False
                frozen += param.numel()
            else:
                param.requires_grad = True
                trainable += param.numel()
        
        logger.info(f"Frozen parameters: {frozen:,}")
        logger.info(f"Trainable parameters: {trainable:,}")
    
    def _log_trainable_params(self) -> None:
        """Log which parameters are trainable."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Trainable: {name} ({param.numel():,} params)")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer for router training."""
        train_config = self.config.get("training", {}).get("router_train", {})
        
        lr = float(train_config.get("learning_rate", 1e-3))
        weight_decay = float(train_config.get("weight_decay", 0.0))
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
        warmup_steps = int(train_config.get("warmup_steps", 1000))
        max_steps = int(train_config.get("max_steps", 50000))
        min_lr = float(train_config.get("min_lr", 1e-4))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(min_lr / lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for router training."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        
        # Forward pass - model returns aux_loss from MoE layers
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            return_aux_loss=True,
        )
        
        # The loss includes both LM loss and aux loss
        # For router training, we want the aux_loss to be the primary signal
        if outputs.get("aux_loss") is not None:
            outputs["total_loss"] = outputs.get("loss", 0) + outputs["aux_loss"]
        else:
            outputs["total_loss"] = outputs.get("loss", 0)
        
        return outputs
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with detailed routing stats logging."""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        
        # Forward pass with autocast
        with self.autocast:
            outputs = self.compute_loss(batch)
            loss = outputs.get("total_loss", outputs.get("loss"))
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.accum_step += 1
        
        # Optimizer step
        if self.accum_step >= self.gradient_accumulation_steps:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            self.optimizer.zero_grad()
            self.accum_step = 0
            self.state.step += 1
        
        # Update tokens seen
        if "input_ids" in batch:
            self.state.tokens_seen += batch["input_ids"].numel()
        
        # Return metrics including aux_loss
        metrics = {"loss": loss.item() * self.gradient_accumulation_steps}
        if "aux_loss" in outputs and outputs["aux_loss"] is not None:
            metrics["aux_loss"] = outputs["aux_loss"].item()
        if "total_loss" in outputs:
            metrics["total_loss"] = outputs["total_loss"].item()
        
        return metrics
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get aggregated routing statistics from all MoE layers."""
        moe_layers = self.moe_model.get_moe_layers()
        
        all_stats = {}
        for i, layer in enumerate(moe_layers):
            stats = layer.get_routing_stats() if hasattr(layer, "get_routing_stats") else {}
            for key, value in stats.items():
                if key not in all_stats:
                    all_stats[key] = []
                all_stats[key].append(value)
        
        # Aggregate
        aggregated = {}
        for key, values in all_stats.items():
            if isinstance(values[0], list):
                # Expert utilization per layer
                aggregated[f"{key}_per_layer"] = values
                # Average across layers
                aggregated[f"{key}_mean"] = [
                    sum(v[i] for v in values) / len(values)
                    for i in range(len(values[0]))
                ]
            else:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    def log_routing_stats(self) -> None:
        """Log detailed routing statistics."""
        stats = self.get_routing_stats()
        
        # Log key metrics
        if "expert_utilization_mean" in stats:
            util = stats["expert_utilization_mean"]
            logger.info(
                f"Routing: min_usage={min(util):.2f}% max_usage={max(util):.2f}% "
                f"mean_usage={sum(util)/len(util):.2f}% "
                f"cv={stats.get('load_balance_cv', 0):.4f} "
                f"entropy={stats.get('routing_entropy', 0):.4f} "
                f"dead={stats.get('dead_experts', 0)} "
                f"top1_frac={stats.get('top1_fraction', 0):.2f}% "
                f"top2_frac={stats.get('top2_fraction', 0):.2f}% "
                f"aux_loss={stats.get('aux_loss', 0):.4f}"
            )
        
        return stats
    
    @classmethod
    def from_config(
        cls,
        model: MoELanguageModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> "RouterTrainer":
        """Create trainer from config."""
        return cls(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
        )