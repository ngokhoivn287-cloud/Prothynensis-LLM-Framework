from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class SyntheticExample:
    id: str
    domain: str
    difficulty: str
    task_type: str
    language: str
    text: str


class SharedUniqueDataset:
    def __init__(self, shared_examples: List[SyntheticExample], unique_examples: List[SyntheticExample], tokenizer, max_seq_len: int):
        self.examples = list(shared_examples) + list(unique_examples)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

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
