"""Modular MoE package."""

from .model import (
    MoELanguageModel,
    create_model_from_config,
)
from .training import (
    BaseTrainer,
    ExpertTrainer,
    RouterTrainer,
    JointTrainer,
    CheckpointManager,
)
from .data import (
    TokenizedDataset,
    create_tokenizer,
    prepare_and_shard_dataset,
)
from .evaluation import (
    evaluate_perplexity,
    evaluate_routing_detailed,
)

__version__ = "0.1.0"

__all__ = [
    "MoELanguageModel",
    "create_model_from_config",
    "BaseTrainer",
    "ExpertTrainer",
    "RouterTrainer",
    "JointTrainer",
    "CheckpointManager",
    "TokenizedDataset",
    "create_tokenizer",
    "prepare_and_shard_dataset",
    "evaluate_perplexity",
    "evaluate_routing_detailed",
]