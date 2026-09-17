"""Perplexity evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from tqdm import tqdm

from ..utils.logging import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_autocast: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, float]:
    """Evaluate perplexity on a dataset."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
    
    with autocast_ctx:
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            if max_batches and i >= max_batches:
                break
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass
            input_ids = batch["input_ids"]
            labels = batch.get("labels", input_ids)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.get("loss")
            
            if loss is not None:
                # Count tokens (excluding padding)
                if "attention_mask" in batch:
                    num_tokens = batch["attention_mask"].sum().item()
                else:
                    num_tokens = (labels != -100).sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                
                # Accuracy
                logits = outputs["logits"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                predictions = shift_logits.argmax(dim=-1)
                mask = shift_labels != -100
                correct = (predictions == shift_labels) & mask
                total_correct += correct.sum().item()
    
    metrics = {}
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        metrics["loss"] = avg_loss
        metrics["perplexity"] = torch.exp(torch.tensor(avg_loss)).item()
        metrics["accuracy"] = total_correct / total_tokens if total_tokens > 0 else 0.0
        metrics["total_tokens"] = total_tokens
    
    return metrics


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.1,
) -> List[Dict[str, Any]]:
    """Evaluate generation quality on a set of prompts."""
    model.eval()
    
    results = []
    
    for prompt in prompts:
        # Tokenize prompt
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        
        # Generate
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        
        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())
        
        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "prompt_length": input_ids.size(1),
            "generated_length": generated.size(1) - input_ids.size(1),
        })
    
    return results