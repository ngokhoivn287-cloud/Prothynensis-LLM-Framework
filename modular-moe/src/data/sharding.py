"""Deterministic dataset sharding for modular expert pretraining."""

from __future__ import annotations

import os
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .tokenizer import BaseTokenizer, create_tokenizer

try:
    from ..utils.paths import is_allowed_dataset_path, get_shards_root, get_temp_root, assert_allowed_path
except ImportError:
    def is_allowed_dataset_path(path):
        return True
    def get_shards_root():
        return Path("shards")
    def get_temp_root():
        return Path("temp")
    def assert_allowed_path(path, context=""):
        return Path(path)

try:
    from ..utils.logging import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

try:
    from ..utils.hardware import set_deterministic
except ImportError:
    def set_deterministic(seed):
        pass

logger = get_logger(__name__)


@dataclass
class ShardMetadata:
    """Metadata for a dataset shard."""
    shard_id: int
    path: str
    num_tokens: int
    num_documents: int
    checksum: str
    expert_id: Optional[int] = None
    model_id: Optional[int] = None
    is_shared: bool = False


def compute_checksum(data: np.ndarray) -> str:
    """Compute SHA256 checksum of array."""
    return hashlib.sha256(data.tobytes()).hexdigest()


def save_shard(
    data: np.ndarray,
    path: Path,
    dtype: np.dtype = np.uint16,
) -> str:
    """Save shard data to binary file and return checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data.astype(dtype).tofile(path)
    return compute_checksum(data)


def load_raw_dataset(config: dict) -> List[str]:
    """Load raw text dataset from various sources."""
    dataset_name = config.get("dataset_name", "fineweb")
    dataset_config = config.get("dataset_config", "sample-10BT")
    text_column = config.get("text_column", "text")
    
    if dataset_name == "synthetic":
        # Generate synthetic data for testing
        return generate_synthetic_data(config)
    
    # Try to load from HuggingFace datasets
    try:
        from datasets import load_dataset
        
        logger.info(f"Loading dataset: {dataset_name}/{dataset_config}")
        dataset = load_dataset(dataset_name, dataset_config, split="train", streaming=True)
        
        texts = []
        for item in dataset:
            text = item.get(text_column, "")
            if text and isinstance(text, str):
                texts.append(text)
        
        logger.info(f"Loaded {len(texts)} documents")
        return texts
    
    except ImportError:
        logger.warning("datasets library not available, using synthetic data")
        return generate_synthetic_data(config)
    except Exception as e:
        logger.warning(f"Failed to load {dataset_name}: {e}, using synthetic data")
        return generate_synthetic_data(config)


def generate_synthetic_data(config: dict) -> List[str]:
    """Generate synthetic text data for testing."""
    vocab_size = config.get("tokenizer_vocab_size", 1024)
    num_docs = config.get("synthetic_num_docs", 1000)
    max_doc_len = config.get("max_doc_length", 1000)
    min_doc_len = config.get("min_doc_length", 100)
    
    # Simple synthetic text generation
    import random
    random.seed(config.get("shard_seed", 42))
    
    # Create a small vocabulary of "words"
    words = [f"word_{i}" for i in range(min(1000, vocab_size // 4))]
    
    texts = []
    for _ in range(num_docs):
        doc_len = random.randint(min_doc_len, max_doc_len)
        doc = " ".join(random.choices(words, k=doc_len))
        texts.append(doc)
    
    return texts


def normalize_text(text: str, form: str = "NFKC") -> str:
    """Normalize text."""
    import unicodedata
    return unicodedata.normalize(form, text)


def deduplicate_texts(
    texts: List[str],
    method: str = "minhash",
    threshold: float = 0.9,
) -> List[str]:
    """Deduplicate texts using specified method."""
    if method == "exact":
        # Exact deduplication
        seen = set()
        unique = []
        for text in texts:
            h = hashlib.sha256(text.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(text)
        return unique
    
    elif method == "minhash":
        # MinHash deduplication (approximate)
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning("datasketch not installed, falling back to exact deduplication")
            return deduplicate_texts(texts, method="exact")
        
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        unique = []
        
        for i, text in enumerate(texts):
            # Create MinHash
            m = MinHash(num_perm=128)
            # Simple tokenization for MinHash
            for word in text.split()[:1000]:  # Limit tokens
                m.update(word.encode())
            
            # Check for near-duplicates
            if not lsh.query(m):
                lsh.insert(str(i), m)
                unique.append(text)
        
        return unique
    
    else:
        logger.warning(f"Unknown deduplication method: {method}, skipping")
        return texts


def tokenize_texts(
    texts: List[str],
    tokenizer: BaseTokenizer,
    max_length: Optional[int] = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Tokenize a list of texts into a single flat array."""
    all_tokens = []
    
    iterator = tqdm(texts, desc="Tokenizing") if show_progress else texts
    
    for text in iterator:
        tokens = tokenizer.encode(text)
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]
        all_tokens.extend(tokens)
        # Add EOS token between documents
        all_tokens.append(tokenizer.eos_token_id)
    
    return np.array(all_tokens, dtype=np.uint16)


def shard_dataset(
    tokens: np.ndarray,
    num_shards: int,
    expert_specific_ratio: float = 0.8,
    num_experts: int = 32,
    shard_strategy: str = "round_robin",
    seed: int = 42,
    model_specific_ratio: Optional[float] = None,
    num_models: Optional[int] = None,
) -> List[np.ndarray]:
    """Shard tokenized data deterministically.
    
    Args:
        tokens: Flat array of token IDs
        num_shards: Number of shards to create
        expert_specific_ratio: Fraction of data for expert-specific shards (MoE terminology)
        num_experts: Number of experts (MoE terminology)
        shard_strategy: "round_robin", "random", or "sequential"
        seed: Random seed for reproducibility
        model_specific_ratio: Fraction of data for model-specific shards (MoMMs terminology)
        num_models: Number of models (MoMMs terminology)
    
    Returns:
        List of shard arrays
    """
    # MoMMs terminology aliases
    if model_specific_ratio is not None:
        expert_specific_ratio = model_specific_ratio
    if num_models is not None:
        num_experts = num_models
    set_deterministic(seed)
    rng = np.random.RandomState(seed)
    
    total_tokens = len(tokens)
    
    # Calculate shard sizes
    if shard_strategy == "sequential":
        # Simple sequential split
        shard_size = total_tokens // num_shards
        shards = []
        for i in range(num_shards):
            start = i * shard_size
            end = start + shard_size if i < num_shards - 1 else total_tokens
            shards.append(tokens[start:end])
        return shards
    
    # For round_robin or random, we need to handle expert-specific + shared
    # Expert-specific shards: one per expert
    # Shared shard: remaining data
    
    num_expert_shards = min(num_experts, num_shards)
    num_shared_shards = num_shards - num_expert_shards
    
    # Allocate tokens
    expert_tokens_count = int(total_tokens * expert_specific_ratio)
    shared_tokens_count = total_tokens - expert_tokens_count
    
    # Split expert tokens among expert shards
    expert_shard_size = expert_tokens_count // num_expert_shards
    shared_shard_size = shared_tokens_count // max(1, num_shared_shards)
    
    if shard_strategy == "round_robin":
        # Round-robin distribution for expert shards
        expert_shards = [[] for _ in range(num_expert_shards)]
        
        for i in range(expert_tokens_count):
            shard_idx = i % num_expert_shards
            expert_shards[shard_idx].append(tokens[i])
        
        # Convert to arrays
        shards = [np.array(s, dtype=np.uint16) for s in expert_shards]
        
        # Shared shards (sequential from remaining tokens)
        shared_start = expert_tokens_count
        for i in range(num_shared_shards):
            start = shared_start + i * shared_shard_size
            end = start + shared_shard_size if i < num_shared_shards - 1 else total_tokens
            if start < total_tokens:
                shards.append(tokens[start:end])
        
        return shards
    
    elif shard_strategy == "random":
        # Random permutation then sequential
        indices = rng.permutation(total_tokens)
        shuffled_tokens = tokens[indices]
        
        return shard_dataset(
            shuffled_tokens, num_shards, expert_specific_ratio, num_experts,
            shard_strategy="sequential", seed=seed
        )
    
    else:
        raise ValueError(f"Unknown shard strategy: {shard_strategy}")


def prepare_and_shard_dataset(
    config: dict,
    output_dir: str,
    tokenizer: Optional[BaseTokenizer] = None,
) -> List[ShardMetadata]:
    """Full pipeline: load, preprocess, tokenize, shard, and save."""
    
    output_path = Path(output_dir)
    if not is_allowed_dataset_path(output_path):
        output_path = get_shards_root() / config.get("data", {}).get("dataset_name", "synthetic")
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Load config
    data_config = config.get("data", config)
    
    # Create tokenizer if not provided
    if tokenizer is None:
        tokenizer = create_tokenizer(data_config)
    
    # Load raw data
    texts = load_raw_dataset(data_config)
    
    # Normalize
    if data_config.get("normalize", True):
        norm_form = data_config.get("normalization_form", "NFKC")
        texts = [normalize_text(t, norm_form) for t in texts]
    
    # Deduplicate
    if data_config.get("deduplicate", False):
        method = data_config.get("dedup_method", "minhash")
        threshold = data_config.get("dedup_threshold", 0.9)
        texts = deduplicate_texts(texts, method, threshold)
    
    # Tokenize
    max_doc_len = data_config.get("max_doc_length")
    tokens = tokenize_texts(texts, tokenizer, max_doc_len)
    
    logger.info(f"Total tokens: {len(tokens):,}")
    
    # Shard
    num_shards = data_config.get("num_shards", 32)
    expert_specific_ratio = data_config.get("expert_specific_ratio", 0.8)
    num_experts = data_config.get("num_experts", 32)
    shard_strategy = data_config.get("shard_strategy", "round_robin")
    shard_seed = data_config.get("shard_seed", 42)
    
    # MoMMs terminology aliases
    model_specific_ratio = data_config.get("model_specific_ratio", expert_specific_ratio)
    num_models = data_config.get("num_models", num_experts)
    
    shards = shard_dataset(
        tokens, num_shards, expert_specific_ratio, num_experts,
        shard_strategy, shard_seed,
        model_specific_ratio=model_specific_ratio,
        num_models=num_models,
    )
    
    # Save shards
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    for i, shard in enumerate(shards):
        shard_path = output_path / f"shard_{i:05d}.bin"
        checksum = save_shard(shard, shard_path)
        
        # Assign expert/model IDs
        if i < min(num_experts, num_shards):
            expert_id = i
            model_id = i
            is_shared = False
        else:
            expert_id = None
            model_id = None
            is_shared = True
        
        meta = ShardMetadata(
            shard_id=i,
            path=shard_path.name,
            num_tokens=len(shard),
            num_documents=0,
            checksum=checksum,
            expert_id=expert_id,
            model_id=model_id,
            is_shared=is_shared,
        )
        metadata.append(meta)
    
    # Save metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump([asdict(m) for m in metadata], f, indent=2)
    
    logger.info(f"Saved {len(shards)} shards to {output_dir}")
    
    return metadata


def verify_shards(shard_dir: str) -> bool:
    """Verify all shards match their checksums."""
    shard_path = Path(shard_dir)
    if not is_allowed_dataset_path(shard_path):
        logger.error(f"Refusing to verify shards outside approved roots: {shard_path}")
        return False
    metadata_path = shard_path / "metadata.json"
    if not metadata_path.exists():
        return False
    
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    all_ok = True
    for meta in metadata:
        path = Path(shard_dir) / meta["path"]
        if not path.exists():
            logger.error(f"Missing shard: {path}")
            all_ok = False
            continue
        
        data = np.fromfile(path, dtype=np.uint16)
        checksum = compute_checksum(data)
        
        if checksum != meta["checksum"]:
            logger.error(f"Checksum mismatch for {path}: {checksum} != {meta['checksum']}")
            all_ok = False
    
    if all_ok:
        logger.info("All shards verified successfully")
    
    return all_ok


def get_shard_for_expert(
    shard_dir: str,
    expert_id: int,
    include_shared: bool = True,
) -> List[np.ndarray]:
    """Load shard(s) assigned to a specific expert/model."""
    shard_path = Path(shard_dir)
    if not is_allowed_dataset_path(shard_path):
        raise PermissionError(f"Refusing to load shards from outside approved roots: {shard_path}")
    metadata_path = shard_path / "metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    shards = []
    for meta in metadata:
        if meta.get("expert_id") == expert_id or meta.get("model_id") == expert_id or (include_shared and meta["is_shared"]):
            path = Path(shard_dir) / meta["path"]
            data = np.fromfile(path, dtype=np.uint16)
            shards.append(data)
    
    return shards


def get_shard_for_model(
    shard_dir: str,
    model_id: int,
    include_shared: bool = True,
) -> List[np.ndarray]:
    """Load shard(s) assigned to a specific model (MoMMs terminology)."""
    return get_shard_for_expert(shard_dir, model_id, include_shared)