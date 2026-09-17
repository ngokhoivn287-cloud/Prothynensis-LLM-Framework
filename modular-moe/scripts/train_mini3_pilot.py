from __future__ import annotations

import argparse
import random
from typing import Dict, List, Optional

import torch

from src.data.tokenizer import CharacterTokenizer
from src.model import MoELanguageModel
from src.training.training_controller import QualityLock


class SharedUniqueDataset:
    def __init__(self, shared_examples, unique_examples, tokenizer, max_seq_len: int, shared_ratio: float = 0.1):
        self.examples = list(shared_examples) + list(unique_examples)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shared_ratio = shared_ratio

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        ids = self.tokenizer.encode(example.text)
        input_ids = torch.full((self.max_seq_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        labels = torch.full((self.max_seq_len,), self.tokenizer.pad_token_id, dtype=torch.long)
        length = min(len(ids), self.max_seq_len)
        if length > 0:
            input_ids[:length] = torch.tensor(ids[:length], dtype=torch.long)
            labels[:length] = torch.tensor(ids[:length], dtype=torch.long)
        if length == 0:
            labels[0] = self.tokenizer.eos_token_id
        return {"input_ids": input_ids, "labels": labels}

    def get_shared_ratio(self) -> float:
        return self.shared_ratio


def _build_synthetic_dataset(num_docs: int, seed: int, domain_weights: Optional[Dict[str, float]] = None) -> List:
    random.seed(seed)
    examples = []
    domains = ["math", "coding", "reasoning", "language"]
    weights = [1.0] * len(domains)
    if domain_weights:
        for i, domain in enumerate(domains):
            weights[i] = domain_weights.get(domain, 1.0)
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    for i in range(num_docs):
        domain = random.choices(domains, weights=weights, k=1)[0]
        if domain == "math":
            text = f"Q: What is {i}+{i}? A: {2*i} (math doc {i})"
        elif domain == "coding":
            text = f"Q: How do you print in Python? A: print('hello {i}') (coding doc {i})"
        elif domain == "reasoning":
            text = f"Q: If all A are B and all B are C, what can we conclude? A: All A are C. (reasoning doc {i})"
        else:
            text = f"Q: Explain concept {i}. A: Concept {i} is about understanding. (language doc {i})"
        examples.append(
            type("SyntheticExample", (), {
                "id": f"e{i}",
                "domain": domain,
                "difficulty": "D0",
                "task_type": domain,
                "language": "en",
                "text": text,
            })()
        )
    return examples


def build_model_config(vocab_size: int = 1024, max_seq_len: int = 128) -> Dict:
    return {
        "vocab_size": vocab_size,
        "max_seq_len": max_seq_len,
        "hidden_dim": 512,
        "num_layers": 12,
        "num_heads": 8,
        "head_dim": 64,
        "num_experts": 1,
        "top_k": 1,
        "expert_hidden_dim": 2048,
        "expert_activation": "silu",
        "expert_norm": "rmsnorm",
        "expert_bias": False,
        "expert_dropout": 0.0,
        "router_hidden_dim": None,
        "router_bias": False,
        "router_dropout": 0.0,
        "tie_embeddings": True,
        "use_rmsnorm": True,
        "norm_eps": 1e-5,
        "attn_dropout": 0.0,
        "resid_dropout": 0.0,
        "causal": True,
        "use_flash_attn": False,
        "gradient_checkpointing": False,
    }


def build_model(config: Dict, device):
    model = MoELanguageModel(**config)
    model.to(device)
    return model


def count_parameters(model) -> Dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def build_parser():
    parser = argparse.ArgumentParser(description="Mini3 Pilot Training")
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    return parser


def run_pilot(args=None):
    parser = build_parser()
    parsed = parser.parse_args(args)
    config = build_model_config(vocab_size=parsed.vocab_size, max_seq_len=parsed.max_seq_len)
    device = torch.device("cpu")
    model = build_model(config, device)
    params = count_parameters(model)
    print(f"Model parameters: total={params['total']}, trainable={params['trainable']}")
