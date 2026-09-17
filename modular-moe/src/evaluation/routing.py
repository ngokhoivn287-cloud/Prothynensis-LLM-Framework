"""Routing and expert utilization evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Optional
from collections import defaultdict
import numpy as np

from ..utils.logging import get_logger
from ..model import MoELanguageModel, MoELayer

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_routing(
    model: MoELanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_autocast: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Evaluate routing statistics across all MoE layers."""
    model.eval()
    
    moe_layers = model.get_moe_layers()
    if not moe_layers:
        return {"error": "No MoE layers found"}
    
    # Reset stats
    for layer in moe_layers:
        if hasattr(layer, "reset_stats"):
            layer.reset_stats()
    
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
    
    with autocast_ctx:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with aux loss to trigger routing
            _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                return_aux_loss=True,
            )
    
    # Aggregate stats across layers
    all_stats = defaultdict(list)
    
    for layer_idx, layer in enumerate(moe_layers):
        if hasattr(layer, "get_routing_stats"):
            # Need router logits and indices - we need to capture these during forward
            pass
    
    # Instead, let's do a more detailed evaluation pass
    return evaluate_routing_detailed(model, dataloader, device, max_batches, use_autocast, dtype)


@torch.no_grad()
def evaluate_routing_detailed(
    model: MoELanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    use_autocast: bool = True,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Detailed routing evaluation with per-layer statistics."""
    model.eval()
    
    moe_layers = model.get_moe_layers()
    num_layers = len(moe_layers)
    
    if num_layers == 0:
        return {"error": "No MoE layers found"}
    
    num_experts = moe_layers[0].num_experts
    top_k = moe_layers[0].top_k
    
    # Accumulators
    expert_token_counts = [torch.zeros(num_experts, device=device) for _ in range(num_layers)]
    router_entropies = [[] for _ in range(num_layers)]
    all_router_logits = []
    
    autocast_ctx = torch.autocast(device_type=device.type, dtype=dtype) if use_autocast else torch.no_grad()
    
    with autocast_ctx:
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break
            
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # We need to hook into the model to capture routing decisions
            # For now, do a forward pass that returns routing info
            routing_info = capture_routing_info(model, batch)
            
            if routing_info:
                for layer_idx, info in enumerate(routing_info):
                    if info is None:
                        continue
                    
                    expert_indices = info["expert_indices"]  # [num_tokens, top_k]
                    router_logits = info["router_logits"]    # [num_tokens, num_experts]
                    
                    # Count tokens per expert
                    flat_indices = expert_indices.flatten()
                    counts = torch.bincount(flat_indices, minlength=num_experts)
                    expert_token_counts[layer_idx] += counts
                    
                    # Routing entropy
                    router_probs = torch.softmax(router_logits, dim=-1)
                    entropy = -(router_probs * torch.log(router_probs + 1e-10)).sum(dim=-1).mean()
                    router_entropies[layer_idx].append(entropy.item())
    
    # Compute metrics
    results = {
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "layers": [],
    }
    
    total_tokens_all_layers = 0
    all_expert_utils = []
    
    for layer_idx in range(num_layers):
        counts = expert_token_counts[layer_idx]
        total_tokens = counts.sum().item()
        total_tokens_all_layers += total_tokens
        
        if total_tokens == 0:
            layer_stats = {
                "layer": layer_idx,
                "total_tokens": 0,
                "expert_utilization": [0.0] * num_experts,
                "routing_entropy": 0.0,
                "dead_experts": num_experts,
                "max_utilization": 0.0,
                "min_utilization": 0.0,
                "load_balance_cv": 0.0,
            }
        else:
            utilization = (counts.float() / (total_tokens * top_k) * 100).cpu().numpy()
            mean_load = counts.float().mean().item()
            std_load = counts.float().std().item()
            cv = std_load / (mean_load + 1e-10)
            
            dead_experts = (utilization < 1.0).sum()
            max_util = utilization.max()
            min_util = utilization.min()
            
            avg_entropy = np.mean(router_entropies[layer_idx]) if router_entropies[layer_idx] else 0.0
            
            layer_stats = {
                "layer": layer_idx,
                "total_tokens": total_tokens,
                "expert_utilization": utilization.tolist(),
                "routing_entropy": avg_entropy,
                "dead_experts": int(dead_experts),
                "max_utilization": float(max_util),
                "min_utilization": float(min_util),
                "load_balance_cv": float(cv),
                "expert_token_counts": counts.cpu().tolist(),
            }
        
        results["layers"].append(layer_stats)
        all_expert_utils.append(layer_stats["expert_utilization"])
    
    # Global stats
    if all_expert_utils:
        mean_utils = np.mean(all_expert_utils, axis=0)
        results["global_expert_utilization"] = mean_utils.tolist()
        results["global_dead_experts"] = int((mean_utils < 1.0).sum())
        results["global_max_utilization"] = float(mean_utils.max())
        results["global_min_utilization"] = float(mean_utils.min())
        results["global_routing_entropy"] = float(np.mean([
            l["routing_entropy"] for l in results["layers"]
        ]))
        results["global_load_balance_cv"] = float(np.mean([
            l["load_balance_cv"] for l in results["layers"]
        ]))
    
    return results


def capture_routing_info(model: MoELanguageModel, batch: Dict[str, torch.Tensor]) -> List[Optional[Dict]]:
    """Capture routing information from MoE layers during forward pass."""
    # This would require modifying the model to return routing info
    # For now, return None - implement by adding hooks or modifying forward
    return []


@torch.no_grad()
def evaluate_expert_specialization(
    model: MoELanguageModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate expert specialization by analyzing token-type routing."""
    # This would require labeled data or clustering
    # Placeholder for future implementation
    return {"note": "Expert specialization evaluation not yet implemented"}


def compute_routing_metrics(
    expert_token_counts: List[torch.Tensor],
    num_experts: int,
    top_k: int,
) -> Dict[str, Any]:
    """Compute routing metrics from expert token counts."""
    metrics = {}
    
    for layer_idx, counts in enumerate(expert_token_counts):
        total = counts.sum().item()
        if total == 0:
            continue
        
        utilization = counts.float() / (total * top_k) * 100
        
        metrics[f"layer_{layer_idx}"] = {
            "expert_utilization": utilization.tolist(),
            "dead_experts": (utilization < 1.0).sum().item(),
            "overloaded_experts": (utilization > 100.0 / num_experts * 2).sum().item(),
            "max_utilization": utilization.max().item(),
            "min_utilization": utilization.min().item(),
            "load_balance_cv": (counts.float().std() / (counts.float().mean() + 1e-10)).item(),
            "routing_entropy": (-(utilization / 100 * torch.log(utilization / 100 + 1e-10)).sum()).item(),
        }
    
    return metrics