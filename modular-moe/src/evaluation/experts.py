"""Expert evaluation utilities."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..utils.logging import get_logger
from ..model import ExpertFFN

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_expert(
    expert: ExpertFFN,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_autocast: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Evaluate a single expert as a standalone FFN."""
    expert.eval()
    expert.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    output_norms = []
    
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
    
    with autocast_ctx:
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating expert", leave=False)):
            if max_batches and i >= max_batches:
                break
            
            # For expert evaluation, we need input/target pairs
            # This assumes the dataloader provides "input" and "target" or similar
            if "input" in batch:
                x = batch["input"].to(device)
                target = batch.get("target", x).to(device)
            else:
                # Use input_ids as proxy
                x = batch["input_ids"].to(device).float()
                target = x
            
            # Forward pass
            output = expert(x)
            
            # Compute loss (MSE for FFN output)
            loss = torch.nn.functional.mse_loss(output, target)
            
            num_tokens = x.numel() // x.size(-1)
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            # Track output norms
            output_norms.append(output.norm(dim=-1).mean().item())
    
    metrics = {}
    if total_tokens > 0:
        metrics["mse_loss"] = total_loss / total_tokens
        metrics["output_norm_mean"] = sum(output_norms) / len(output_norms) if output_norms else 0.0
        metrics["total_tokens"] = total_tokens
    
    return metrics


@torch.no_grad()
def evaluate_expert_language_modeling(
    expert: ExpertFFN,
    dataloader: DataLoader,
    device: torch.device,
    vocab_size: int,
    max_seq_len: int,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate expert by wrapping in a minimal LM."""
    # Build minimal model around expert
    from ..training.expert_trainer import ExpertTrainer
    
    class MinimalLM(nn.Module):
        def __init__(self, expert, vocab_size, max_seq_len, hidden_dim):
            super().__init__()
            self.expert = expert
            self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            self.lm_head.weight = self.token_embedding.weight
        
        def forward(self, input_ids, labels=None):
            batch_size, seq_len = input_ids.shape
            pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = self.token_embedding(input_ids) + self.position_embedding(pos_ids)
            x = self.expert(x)
            logits = self.lm_head(x)
            
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            
            return {"logits": logits, "loss": loss, "total_loss": loss}
    
    hidden_dim = expert.hidden_dim
    model = MinimalLM(expert, vocab_size, max_seq_len, hidden_dim).to(device)
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating expert LM", leave=False)):
            if max_batches and i >= max_batches:
                break
            
            batch = {k: v.to(device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.get("loss")
            
            if loss is not None:
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
    
    metrics = {}
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        metrics["loss"] = avg_loss
        metrics["perplexity"] = torch.exp(torch.tensor(avg_loss)).item()
        metrics["total_tokens"] = total_tokens
    
    return metrics


def compare_experts(
    experts: List[ExpertFFN],
    dataloader: DataLoader,
    device: torch.device,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Compare multiple experts on the same data."""
    results = []
    for i, expert in enumerate(experts):
        logger.info(f"Evaluating expert {i}")
        metrics = evaluate_expert(expert, dataloader, device, **kwargs)
        metrics["expert_id"] = i
        results.append(metrics)
    return results


def check_expert_collapse(
    model,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Check for expert collapse (one expert dominating)."""
    # This would require routing info - placeholder
    return {"note": "Expert collapse detection requires routing evaluation"}