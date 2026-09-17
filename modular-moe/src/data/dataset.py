"""Dataset handling for training and evaluation."""

from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any
from dataclasses import dataclass, asdict

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
import numpy as np
from tqdm import tqdm

try:
    from ..utils.logging import get_logger
except ImportError:
    def get_logger(name):
        import logging
        return logging.getLogger(name)
from .tokenizer import BaseTokenizer, create_tokenizer

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
    is_shared: bool = False


@dataclass
class DatasetConfig:
    """Configuration for a dataset."""
    name: str
    path: str
    split: str = "train"
    tokenizer_name: str = "default"
    max_seq_len: int = 2048
    domain: str = "general"
    difficulty: str = "D2"
    task_type: str = "reasoning"
    language: str = "en"
    quality_score: float = 1.0


class TokenizedDataset(Dataset):
    """Dataset of tokenized sequences."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        stride: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len
        
        # Load data
        self.data = self._load_data()
        self.total_tokens = len(self.data)
        self.num_examples = max(0, (self.total_tokens - max_seq_len) // self.stride + 1)
    
    def _load_data(self) -> torch.Tensor:
        """Load tokenized data from file."""
        if self.data_path.suffix == ".bin":
            # Binary format (uint16 or uint32)
            data = np.fromfile(self.data_path, dtype=np.uint16)
            return torch.from_numpy(data.astype(np.int64))
        elif self.data_path.suffix == ".pt":
            # PyTorch tensor
            return torch.load(self.data_path)
        elif self.data_path.suffix == ".npy":
            # NumPy array
            data = np.load(self.data_path)
            return torch.from_numpy(data.astype(np.int64))
        else:
            raise ValueError(f"Unsupported data format: {self.data_path.suffix}")
    
    def __len__(self) -> int:
        return self.num_examples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_seq_len + 1  # +1 for next token prediction
        
        chunk = self.data[start:end]
        
        if len(chunk) < self.max_seq_len + 1:
            # Pad if needed
            padding = torch.full(
                (self.max_seq_len + 1 - len(chunk),),
                self.tokenizer.pad_token_id,
                dtype=torch.long
            )
            chunk = torch.cat([chunk, padding])
        
        input_ids = chunk[:-1]
        labels = chunk[1:]
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class StreamingTokenizedDataset(IterableDataset):
    """Streaming dataset for large tokenized files."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        buffer_size: int = 10000,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Load memory-mapped
        self.data = np.lib.format.open_memmap(self.data_path, mode="r")
        self.total_tokens = len(self.data)
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        # Create generator with worker info
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start_idx = 0
            end_idx = self.total_tokens
        else:
            per_worker = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per_worker
            end_idx = start_idx + per_worker
        
        # Generate examples
        rng = np.random.RandomState(self.seed + worker_info.id if worker_info else self.seed)
        
        if self.shuffle:
            # Generate random start positions
            max_start = end_idx - self.max_seq_len - 1
            if max_start <= start_idx:
                return
            
            while True:
                start = rng.randint(start_idx, max_start)
                end = start + self.max_seq_len + 1
                chunk = self.data[start:end]
                
                input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
                labels = torch.from_numpy(chunk[1:].astype(np.int64))
                
                yield {"input_ids": input_ids, "labels": labels}
        else:
            # Sequential
            for start in range(start_idx, end_idx - self.max_seq_len, self.max_seq_len):
                end = start + self.max_seq_len + 1
                chunk = self.data[start:end]
                
                input_ids = torch.from_numpy(chunk[:-1].astype(np.int64))
                labels = torch.from_numpy(chunk[1:].astype(np.int64))
                
                yield {"input_ids": input_ids, "labels": labels}


class ShardedDataset:
    """Manages multiple dataset shards."""
    
    def __init__(
        self,
        shard_dir: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        streaming: bool = True,
        buffer_size: int = 10000,
    ):
        self.shard_dir = Path(shard_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.streaming = streaming
        self.buffer_size = buffer_size
        
        # Load metadata
        self.metadata = self._load_metadata()
        self.shards = self._load_shards()
    
    def _load_metadata(self) -> List[ShardMetadata]:
        """Load shard metadata from JSON."""
        metadata_path = self.shard_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            data = json.load(f)
        
        return [ShardMetadata(**m) for m in data]
    
    def _load_shards(self) -> List[Dataset]:
        """Load all shards as datasets."""
        shards = []
        for meta in self.metadata:
            path = self.shard_dir / meta.path
            if self.streaming:
                shard = StreamingTokenizedDataset(
                    str(path), self.tokenizer, self.max_seq_len, self.buffer_size
                )
            else:
                shard = TokenizedDataset(
                    str(path), self.tokenizer, self.max_seq_len
                )
            shards.append(shard)
        return shards
    
    def get_shard(self, shard_id: int) -> Dataset:
        """Get a specific shard by ID."""
        for i, meta in enumerate(self.metadata):
            if meta.shard_id == shard_id:
                return self.shards[i]
        raise ValueError(f"Shard {shard_id} not found")
    
    def get_shard_for_expert(self, expert_id: int) -> Dataset:
        """Get the shard assigned to an expert."""
        for i, meta in enumerate(self.metadata):
            if meta.expert_id == expert_id and not meta.is_shared:
                return self.shards[i]
        raise ValueError(f"No shard found for expert {expert_id}")
    
    def get_shared_shard(self) -> Optional[Dataset]:
        """Get the shared data shard."""
        for i, meta in enumerate(self.metadata):
            if meta.is_shared:
                return self.shards[i]
        return None
    
    def combine_shards(self, shard_ids: List[int]) -> Dataset:
        """Combine multiple shards into one dataset."""
        from torch.utils.data import ConcatDataset
        selected = [self.shards[i] for i in shard_ids if i < len(self.shards)]
        return ConcatDataset(selected)
    
    def total_tokens(self) -> int:
        return sum(m.num_tokens for m in self.metadata)
    
    def __len__(self) -> int:
        return len(self.shards)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader with appropriate settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for tokenized sequences."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {"input_ids": input_ids, "labels": labels}