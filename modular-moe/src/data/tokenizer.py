"""Tokenizers for the framework."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, List, Union

import torch
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import PreTrainedTokenizerFast


class BaseTokenizer:
    """Base tokenizer interface."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    
    def encode(self, text: str) -> List[int]:
        raise NotImplementedError
    
    def decode(self, ids: List[int]) -> str:
        raise NotImplementedError
    
    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]
    
    def decode_batch(self, ids_list: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in ids_list]
    
    def __call__(
        self,
        text: Union[str, List[str]],
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
    ) -> dict:
        if isinstance(text, str):
            text = [text]
        
        encoded = self.encode_batch(text)
        
        if truncation and max_length:
            encoded = [ids[:max_length] for ids in encoded]
        
        if padding:
            max_len = max(len(ids) for ids in encoded)
            if max_length:
                max_len = min(max_len, max_length)
            encoded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
        
        result = {"input_ids": encoded}
        
        if return_tensors == "pt":
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        
        return result
    
    @property
    def pad_token_id(self) -> int:
        return 0
    
    @property
    def eos_token_id(self) -> int:
        return 1
    
    @property
    def bos_token_id(self) -> int:
        return 2
    
    @property
    def unk_token_id(self) -> int:
        return 3


class CharacterTokenizer(BaseTokenizer):
    """Simple character-level tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 256):
        super().__init__(vocab_size)
        # Build char vocab
        self.char_to_id = {chr(i): i for i in range(min(vocab_size, 256))}
        self.id_to_char = {i: chr(i) for i in range(min(vocab_size, 256))}
        # Special tokens
        self.char_to_id["<pad>"] = 0
        self.char_to_id["<eos>"] = 1
        self.char_to_id["<bos>"] = 2
        self.char_to_id["<unk>"] = 3
        self.id_to_char[0] = "<pad>"
        self.id_to_char[1] = "<eos>"
        self.id_to_char[2] = "<bos>"
        self.id_to_char[3] = "<unk>"
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.unk_token_id) for c in text]
    
    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, "<unk>") for i in ids)


class HFTokenizerWrapper(BaseTokenizer):
    """Wrapper for HuggingFace tokenizers."""
    
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        vocab_size: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
    ):
        if tokenizer_path and os.path.exists(tokenizer_path):
            # Load from local path
            hf_tokenizer = HFTokenizer.from_file(tokenizer_path)
        else:
            # Load from HF hub
            from transformers import AutoTokenizer
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            # Convert to fast tokenizer if needed
            if hasattr(hf_tokenizer, "_tokenizer"):
                hf_tokenizer = hf_tokenizer._tokenizer
        
        self.tokenizer = hf_tokenizer
        actual_vocab_size = hf_tokenizer.get_vocab_size()
        
        if vocab_size and vocab_size > actual_vocab_size:
            # Pad vocab size
            self.vocab_size = vocab_size
        else:
            self.vocab_size = actual_vocab_size
        
        # Set special tokens
        self._pad_token_id = getattr(hf_tokenizer, "pad_token_id", 0)
        self._eos_token_id = getattr(hf_tokenizer, "eos_token_id", 1)
        self._bos_token_id = getattr(hf_tokenizer, "bos_token_id", 2)
        self._unk_token_id = getattr(hf_tokenizer, "unk_token_id", 3)
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)
    
    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id
    
    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id
    
    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id
    
    @property
    def unk_token_id(self) -> int:
        return self._unk_token_id
    
    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        self.tokenizer.save(path)


def create_tokenizer(config: dict) -> BaseTokenizer:
    """Factory function to create tokenizer from config."""
    tokenizer_type = config.get("tokenizer_type", config.get("tokenizer_name", "hf"))
    
    if tokenizer_type in ("character", "char"):
        return CharacterTokenizer(config.get("tokenizer_vocab_size", 256))
    else:
        return HFTokenizerWrapper(
            tokenizer_name=config.get("tokenizer_name", "gpt2"),
            vocab_size=config.get("tokenizer_vocab_size"),
            tokenizer_path=config.get("tokenizer_path"),
        )


def train_tokenizer(
    texts: List[str],
    vocab_size: int,
    save_path: str,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None,
) -> HFTokenizerWrapper:
    """Train a BPE tokenizer on the given texts."""
    if special_tokens is None:
        special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
    
    # Initialize tokenizer
    tokenizer = HFTokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    
    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )
    
    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    
    return HFTokenizerWrapper(tokenizer_path=save_path)