"""Joint training trainer (Stage F)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, List

from .base_trainer import BaseTrainer
from ..model import MoELanguageModel, MoELayer
from ..utils.logging import get_logger

logger = get_logger(__name__)


class JointTrainer(BaseTrainer):
    """Trainer for joint training (Stage F).
    
    Unfreezes all components and continues normal MoE pretraining.
    Supports progressive unfreezing schedule.
    """
    
    def __init__(
        self,
        model: MoELanguageModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints/global",
        log_dir: str = "logs/joint",
    ):
        self.moe_model = model
        self.unfreeze_schedule = config.get("training", {}).get("joint_train", {}).get("unfreeze_schedule", {})
        self.current_phase = "router_only"
        
        # Initially freeze according to schedule
        self._apply_unfreeze_schedule(0)
        
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
            device=device,
            output_dir=output_dir,
            log_dir=log_dir,
            phase="joint_train",
        )
        
        logger.info("Joint trainer initialized")
        self._log_trainable_params()
    
    def _apply_unfreeze_schedule(self, step: int) -> None:
        """Apply unfreezing schedule based on current step."""
        # Parse schedule
        phases = []
        for step_str, phase in self.unfreeze_schedule.items():
            try:
                step_num = int(step_str)
            except ValueError:
                if step_str.startswith("step_"):
                    step_num = int(step_str[5:])
                else:
                    continue
            phases.append((step_num, phase))
        
        phases.sort()
        
        # Determine current phase
        new_phase = "router_only"
        for step_num, phase in phases:
            if step >= step_num:
                new_phase = phase
            else:
                break
        
        if new_phase != self.current_phase:
            logger.info(f"Step {step}: Unfreezing phase change: {self.current_phase} -> {new_phase}")
            self.current_phase = new_phase
            self._set_requires_grad(new_phase)
    
    def _set_requires_grad(self, phase: str) -> None:
        """Set requires_grad based on phase."""
        for name, param in self.model.named_parameters():
            if phase == "router_only":
                # Only router trainable
                param.requires_grad = "router" in name.lower()
            elif phase == "router + shared":
                # Router + shared (attention, norms, embeddings) trainable
                param.requires_grad = (
                    "router" in name.lower() or
                    "attention" in name.lower() or
                    "attn" in name.lower() or
                    "norm" in name.lower() or
                    "embedding" in name.lower() or
                    "lm_head" in name.lower()
                )
            elif phase == "all":
                # Everything trainable
                param.requires_grad = True
            else:
                param.requires_grad = False
    
    def _log_trainable_params(self) -> None:
        """Log trainable parameter counts."""
        total = 0
        trainable = 0
        for name, param in self.model.named_parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
                logger.debug(f"Trainable: {name} ({param.numel():,})")
        
        logger.info(f"Total parameters: {total:,}")
        logger.info(f"Trainable parameters: {trainable:,} ({100*trainable/total:.1f}%)")
    
    def setup_optimizer(self) -> None:
        """Setup optimizer for joint training."""
        train_config = self.config.get("training", {}).get("joint_train", {})
        
        lr = float(train_config.get("learning_rate", 1.5e-4))
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
        warmup_steps = int(train_config.get("warmup_steps", 5000))
        max_steps = int(train_config.get("max_steps", 500000))
        min_lr = float(train_config.get("min_lr", 1.5e-5))
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return max(min_lr / lr, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159))).item())
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for joint training."""
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        
        outputs = self.model(
            input_ids=input_ids,
            labels=labels,
            return_aux_loss=True,
        )
        
        return outputs
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training step with unfreeze schedule check."""
        # Check unfreeze schedule
        self._apply_unfreeze_schedule(self.state.step)
        
        # If phase changed, need to recreate optimizer with new trainable params
        if hasattr(self, "_last_phase") and self._last_phase != self.current_phase:
            logger.info("Phase changed, recreating optimizer")
            self.setup_optimizer()
        self._last_phase = self.current_phase
        
        return super().train_step(batch)
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get aggregated routing statistics from all MoE layers."""
        moe_layers = self.moe_model.get_moe_layers()
        
        all_stats = {}
        for i, layer in enumerate(moe_layers):
            if hasattr(layer, "get_routing_stats"):
                stats = layer.get_routing_stats()
                for key, value in stats.items():
                    if key not in all_stats:
                        all_stats[key] = []
                    all_stats[key].append(value)
        
        # Aggregate
        aggregated = {}
        for key, values in all_stats.items():
            if isinstance(values[0], list):
                aggregated[f"{key}_per_layer"] = values
                aggregated[f"{key}_mean"] = [
                    sum(v[i] for v in values) / len(values)
                    for i in range(len(values[0]))
                ]
            else:
                aggregated[key] = sum(values) / len(values)
        
        return aggregated
    
    @classmethod
    def from_config(
        cls,
        model: MoELanguageModel,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader],
        config: Dict[str, Any],
    ) -> "JointTrainer":
        """Create trainer from config."""
        return cls(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            config=config,
        )