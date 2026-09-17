"""Base trainer class."""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Iterator
from dataclasses import dataclass
from contextlib import nullcontext

from ..utils.logging import get_logger, TrainingLogger
from ..utils.hardware import get_device, estimate_model_memory, print_memory_estimate
from .checkpoint import CheckpointManager, CheckpointMetadata

logger = get_logger(__name__)


@dataclass
class TrainingState:
    """Current training state."""
    step: int = 0
    epoch: int = 0
    tokens_seen: int = 0
    dataset_position: int = 0
    best_metric: float = float("inf")


class BaseTrainer:
    """Base trainer with common functionality."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "checkpoints",
        log_dir: str = "logs",
        phase: Optional[str] = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        self.device = device or get_device(self.config.get("hardware", {}).get("device", "cuda"))
        self.output_dir = output_dir
        self.log_dir = log_dir
        
        # Training config - use phase-specific sub-config if provided
        train_config = self.config.get("training", {})
        if phase:
            train_config = train_config.get(phase, train_config)
        self.max_steps = train_config.get("max_steps", 100000)
        self.grad_clip = train_config.get("grad_clip", 1.0)
        self.log_every = train_config.get("log_every", 10)
        self.eval_every = train_config.get("eval_every", 1000)
        self.save_every = train_config.get("save_every", 5000)
        self.gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
        
        # Mixed precision
        self.mixed_precision = train_config.get("mixed_precision", True)
        self.precision = train_config.get("precision", "bf16")
        
        # Setup precision
        self._setup_precision()
        
        # Move model to device
        self.model.to(self.device)
        
        # Optimizer and scheduler (to be set by subclasses)
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        
        # Checkpoint manager
        checkpoint_config = self.config.get("checkpoint", {})
        self.checkpoint_manager = CheckpointManager(
            root_dir=output_dir,
            format=checkpoint_config.get("format", "safetensors"),
            keep_last_n=checkpoint_config.get("keep_last_n", 3),
            save_optimizer=checkpoint_config.get("save_optimizer", True),
            save_scheduler=checkpoint_config.get("save_scheduler", True),
            save_rng=checkpoint_config.get("save_rng", True),
        )
        
        # Logging
        self.logger = TrainingLogger("trainer", log_dir)
        
        # State
        self.state = TrainingState()
        self.start_time = time.time()
        
        # Gradient accumulation counter
        self.accum_step = 0
    
    def _setup_precision(self) -> None:
        """Setup mixed precision training."""
        if self.precision == "fp16":
            self.dtype = torch.float16
        elif self.precision == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        
        if self.mixed_precision and self.dtype != torch.float32:
            self.autocast = torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
            )
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.dtype == torch.float16))
        else:
            self.autocast = nullcontext()
            self.scaler = None

    def compute_self_verification_signals(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute self-verification training signals from a batch."""
        signals = {
            "assumption_confidence": 0.5,
            "evidence_quality": 0.5,
            "contradiction_detected": 0.0,
            "uncertainty": 0.5,
            "self_correction_quality": 0.5,
        }
        if "input_ids" not in batch:
            return signals
        input_ids = batch["input_ids"]
        if hasattr(self.model, "forward"):
            try:
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids[:1])
                    logits = outputs.get("logits", outputs)
                    if hasattr(logits, "float"):
                        logits = logits.float()
                    probs = torch.softmax(logits, dim=-1)
                    max_prob = probs.max(dim=-1).values.mean().item()
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
                    signals["assumption_confidence"] = max_prob
                    signals["uncertainty"] = min(1.0, entropy / 10.0)
                    signals["evidence_quality"] = max_prob
            except Exception:
                pass
        return signals
    
    def setup_optimizer(self) -> None:
        """Setup optimizer and scheduler. Override in subclasses."""
        raise NotImplementedError
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch. Override in subclasses."""
        raise NotImplementedError
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
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
        
        # Return metrics
        metrics = {"loss": loss.item() * self.gradient_accumulation_steps}
        if "aux_loss" in outputs and outputs["aux_loss"] is not None:
            metrics["aux_loss"] = outputs["aux_loss"].item()
        if "total_loss" in outputs:
            metrics["total_loss"] = outputs["total_loss"].item()
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        if self.eval_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        total_aux_loss = 0.0
        num_batches = 0
        
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            
            with self.autocast:
                outputs = self.compute_loss(batch)
                loss = outputs.get("loss")
                if loss is not None:
                    total_loss += loss.item()
                
                aux_loss = outputs.get("aux_loss")
                if aux_loss is not None:
                    total_aux_loss += aux_loss.item()
            
            num_batches += 1
        
        metrics = {}
        if num_batches > 0:
            metrics["eval_loss"] = total_loss / num_batches
            metrics["eval_perplexity"] = torch.exp(torch.tensor(metrics["eval_loss"])).item()
            if total_aux_loss > 0:
                metrics["eval_aux_loss"] = total_aux_loss / num_batches
        
        return metrics
    
    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None) -> Path:
        """Save checkpoint."""
        return self.checkpoint_manager.save(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.state.step,
            epoch=self.state.epoch,
            tokens_seen=self.state.tokens_seen,
            dataset_position=self.state.dataset_position,
            config=self.config,
            metrics=metrics or {},
        )
    
    def load_checkpoint(self, checkpoint_path: str) -> CheckpointMetadata:
        """Load checkpoint."""
        return self.checkpoint_manager.load(
            checkpoint_path=Path(checkpoint_path),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
    
    def train(self) -> None:
        """Main training loop."""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Max steps: {self.max_steps}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        
        # Print memory estimate
        if hasattr(self.model, "get_parameter_count"):
            param_count = self.model.get_parameter_count().get("total", 0)
        else:
            param_count = sum(p.numel() for p in self.model.parameters())
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        estimate = estimate_model_memory(
            num_parameters=param_count,
            trainable_parameters=trainable_params,
            dtype=self.dtype,
            batch_size=self.train_dataloader.batch_size,
            seq_len=self.config.get("model", {}).get("max_seq_len", 2048),
            gradient_checkpointing=self.config.get("hardware", {}).get("gradient_checkpointing", False),
        )
        print_memory_estimate(estimate)
        
        self.model.train()
        
        while self.state.step < self.max_steps:
            self.state.epoch += 1
            
            for batch in self.train_dataloader:
                # Training step
                metrics = self.train_step(batch)
                
                # Logging
                if self.state.step % self.log_every == 0:
                    elapsed = time.time() - self.start_time
                    tokens_per_sec = self.state.tokens_seen / elapsed if elapsed > 0 else 0
                    
                    log_metrics = {
                        **metrics,
                        "step": self.state.step,
                        "epoch": self.state.epoch,
                        "lr": self.optimizer.param_groups[0]["lr"] if self.optimizer else 0,
                        "tokens_per_sec": tokens_per_sec,
                    }
                    self.logger.log_metrics(log_metrics, self.state.step)
                
                # Evaluation
                if self.eval_dataloader and self.state.step % self.eval_every == 0:
                    eval_metrics = self.evaluate()
                    self.logger.log_metrics(eval_metrics, self.state.step)
                    
                    # Track best metric
                    if "eval_loss" in eval_metrics:
                        if eval_metrics["eval_loss"] < self.state.best_metric:
                            self.state.best_metric = eval_metrics["eval_loss"]
                
                # Checkpoint
                if self.state.step % self.save_every == 0:
                    self.save_checkpoint(metrics)
                
                # Check if done
                if self.state.step >= self.max_steps:
                    break
            
            # End of epoch
            if self.state.step >= self.max_steps:
                break
        
        # Final checkpoint
        self.save_checkpoint()
        self.logger.close()
        
        logger.info("Training completed")
    
    def estimate_memory(self) -> None:
        """Estimate and print memory requirements."""
        if hasattr(self.model, "get_parameter_count"):
            param_count = self.model.get_parameter_count().get("total", 0)
        else:
            param_count = sum(p.numel() for p in self.model.parameters())
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        estimate = estimate_model_memory(
            num_parameters=param_count,
            trainable_parameters=trainable_params,
            dtype=self.dtype,
            batch_size=self.train_dataloader.batch_size,
            seq_len=self.config.get("model", {}).get("max_seq_len", 2048),
            gradient_checkpointing=self.config.get("hardware", {}).get("gradient_checkpointing", False),
        )
        print_memory_estimate(estimate)