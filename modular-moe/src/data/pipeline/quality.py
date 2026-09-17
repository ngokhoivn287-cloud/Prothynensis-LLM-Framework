#!/usr/bin/env python
"""
Data Quality Pipeline for Prothynesis Modular MoE
Implements filtering, deduplication, and quality control for training data.
"""

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Iterator, Any, Callable
from collections import Counter

try:
    from datasketch import MinHash, MinHashLSH
    DATASKETCH_AVAILABLE = True
except ImportError:
    DATASKETCH_AVAILABLE = False

import numpy as np
from tqdm import tqdm


@dataclass
class Document:
    """Represents a document with metadata."""
    text: str
    source: str
    language: str = "unknown"
    category: str = "unknown"
    url: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.text)
    
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class QualityConfig:
    """Configuration for quality filtering."""
    # Length filters
    min_doc_length: int = 100
    max_doc_length: Optional[int] = None
    min_word_count: int = 20
    max_word_count: Optional[int] = None
    
    # Quality thresholds
    min_char_diversity: float = 0.15  # Minimum unique chars / total chars
    max_repetition_ratio: float = 0.3  # Maximum repeated n-grams
    min_alphanumeric_ratio: float = 0.5  # Minimum alphanumeric characters
    max_special_char_ratio: float = 0.3  # Maximum special characters
    
    # Language filtering
    allowed_languages: Optional[List[str]] = None
    min_language_confidence: float = 0.8
    
    # Content filtering
    remove_boilerplate: bool = True
    remove_navigation: bool = True
    remove_pii: bool = True
    remove_machine_generated: bool = False  # Can be aggressive
    
    # Deduplication
    dedup_method: str = "minhash"  # "exact", "minhash", "simhash"
    dedup_threshold: float = 0.9
    minhash_num_perm: int = 128
    
    # Normalization
    normalize_unicode: bool = True
    normalization_form: str = "NFKC"
    strip_control_chars: bool = True
    
    # Statistics
    track_stats: bool = True


class DocumentFilter:
    """Filters documents based on quality criteria."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.stats = {
            "total": 0,
            "passed": 0,
            "rejected": Counter(),
        }
    
    def normalize(self, text: str) -> str:
        """Normalize text."""
        if self.config.normalize_unicode:
            text = unicodedata.normalize(self.config.normalization_form, text)
        
        if self.config.strip_control_chars:
            # Remove control characters except newlines and tabs
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        return text
    
    def compute_char_diversity(self, text: str) -> float:
        """Compute character diversity (unique chars / total chars)."""
        if not text:
            return 0.0
        unique = len(set(text))
        return unique / len(text)
    
    def compute_repetition_ratio(self, text: str, n: int = 5) -> float:
        """Compute ratio of repeated n-grams."""
        if len(text) < n * 2:
            return 0.0
        
        ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return 0.0
        
        counts = Counter(ngrams)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        return repeated / len(ngrams)
    
    def compute_alphanumeric_ratio(self, text: str) -> float:
        """Compute ratio of alphanumeric characters."""
        if not text:
            return 0.0
        alnum = sum(1 for c in text if c.isalnum() or c.isspace())
        return alnum / len(text)
    
    def compute_special_char_ratio(self, text: str) -> float:
        """Compute ratio of special/punctuation characters."""
        if not text:
            return 0.0
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return special / len(text)
    
    def detect_boilerplate(self, text: str) -> bool:
        """Detect common boilerplate patterns."""
        boilerplate_patterns = [
            r'cookie policy',
            r'privacy policy',
            r'terms of service',
            r'terms and conditions',
            r'all rights reserved',
            r'copyright \d{4}',
            r'sign up for',
            r'subscribe to',
            r'follow us on',
            r'share this',
            r'related articles',
            r'read more',
            r'click here',
            r'advertisement',
            r'sponsored content',
        ]
        
        text_lower = text.lower()
        matches = sum(1 for pattern in boilerplate_patterns if re.search(pattern, text_lower))
        return matches >= 2  # Multiple boilerplate indicators
    
    def detect_navigation(self, text: str) -> bool:
        """Detect navigation/menu content."""
        nav_patterns = [
            r'^home\s*\|',
            r'menu\s*\n',
            r'\b(home|about|contact|login|register|search)\b\s*\n',
            r'breadcrumb',
            r'sitemap',
            r'navigation',
        ]
        
        text_lower = text.lower()
        matches = sum(1 for pattern in nav_patterns if re.search(pattern, text_lower))
        return matches >= 2
    
    def detect_pii(self, text: str) -> bool:
        """Detect potential PII (basic patterns)."""
        pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Phone
        ]
        
        for pattern in pii_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def detect_machine_generated(self, text: str) -> bool:
        """Heuristic detection of machine-generated text."""
        # Very repetitive, low diversity
        if self.compute_char_diversity(text) < 0.05:
            return True
        
        # Excessive repetition
        if self.compute_repetition_ratio(text) > 0.5:
            return True
        
        # Unusual patterns (e.g., repeated phrases)
        words = text.split()
        if len(words) > 100:
            word_counts = Counter(words)
            most_common = word_counts.most_common(5)
            if most_common and most_common[0][1] / len(words) > 0.1:
                return True
        
        return False
    
    def filter(self, doc: Document) -> bool:
        """Apply all filters to a document. Returns True if document passes."""
        self.stats["total"] += 1
        text = doc.text
        
        # Normalize
        text = self.normalize(text)
        
        # Length checks
        if len(text) < self.config.min_doc_length:
            self.stats["rejected"]["too_short"] += 1
            return False
        
        if self.config.max_doc_length and len(text) > self.config.max_doc_length:
            self.stats["rejected"]["too_long"] += 1
            return False
        
        word_count = len(text.split())
        if word_count < self.config.min_word_count:
            self.stats["rejected"]["too_few_words"] += 1
            return False
        
        if self.config.max_word_count and word_count > self.config.max_word_count:
            self.stats["rejected"]["too_many_words"] += 1
            return False
        
        # Character diversity
        char_div = self.compute_char_diversity(text)
        if char_div < self.config.min_char_diversity:
            self.stats["rejected"]["low_char_diversity"] += 1
            return False
        
        # Repetition
        rep_ratio = self.compute_repetition_ratio(text)
        if rep_ratio > self.config.max_repetition_ratio:
            self.stats["rejected"]["high_repetition"] += 1
            return False
        
        # Alphanumeric ratio
        alnum_ratio = self.compute_alphanumeric_ratio(text)
        if alnum_ratio < self.config.min_alphanumeric_ratio:
            self.stats["rejected"]["low_alphanumeric"] += 1
            return False
        
        # Special char ratio
        special_ratio = self.compute_special_char_ratio(text)
        if special_ratio > self.config.max_special_char_ratio:
            self.stats["rejected"]["high_special_chars"] += 1
            return False
        
        # Boilerplate
        if self.config.remove_boilerplate and self.detect_boilerplate(text):
            self.stats["rejected"]["boilerplate"] += 1
            return False
        
        # Navigation
        if self.config.remove_navigation and self.detect_navigation(text):
            self.stats["rejected"]["navigation"] += 1
            return False
        
        # PII
        if self.config.remove_pii and self.detect_pii(text):
            self.stats["rejected"]["pii"] += 1
            return False
        
        # Machine generated
        if self.config.remove_machine_generated and self.detect_machine_generated(text):
            self.stats["rejected"]["machine_generated"] += 1
            return False
        
        self.stats["passed"] += 1
        return True
    
    def get_stats(self) -> Dict:
        return {
            "total": self.stats["total"],
            "passed": self.stats["passed"],
            "rejected": dict(self.stats["rejected"]),
            "pass_rate": self.stats["passed"] / max(1, self.stats["total"]),
        }


class Deduplicator:
    """Handles document deduplication."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.seen_hashes: Set[str] = set()
        self.lsh = None
        self._minhash_available = DATASKETCH_AVAILABLE
        
        if config.dedup_method == "minhash" and self._minhash_available:
            self.lsh = MinHashLSH(threshold=config.dedup_threshold, num_perm=config.minhash_num_perm)
    
    def compute_hash(self, text: str) -> str:
        """Compute normalized hash for exact deduplication."""
        # Normalize: lowercase, remove extra whitespace, punctuation
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def compute_minhash(self, text: str):
        """Compute MinHash for near-deduplication."""
        if not self._minhash_available:
            raise RuntimeError("MinHash not available - install datasketch")
        m = MinHash(num_perm=self.config.minhash_num_perm)
        # Use word-level shingles
        words = text.lower().split()
        shingles = [' '.join(words[i:i+5]) for i in range(len(words) - 4)]
        for shingle in shingles:
            m.update(shingle.encode())
        return m
    
    def is_duplicate(self, doc: Document) -> bool:
        """Check if document is a duplicate."""
        if self.config.dedup_method == "exact":
            doc_hash = self.compute_hash(doc.text)
            if doc_hash in self.seen_hashes:
                return True
            self.seen_hashes.add(doc_hash)
            return False
        
        elif self.config.dedup_method == "minhash" and self.lsh:
            mh = self.compute_minhash(doc.text)
            # Query LSH for similar documents
            similar = self.lsh.query(mh)
            if similar:
                return True
            # Insert into LSH
            doc_id = f"{doc.source}_{hashlib.md5(doc.text.encode()).hexdigest()[:8]}"
            self.lsh.insert(doc_id, mh)
            return False
        
        return False
    
    def get_stats(self) -> Dict:
        return {
            "method": self.config.dedup_method,
            "unique_documents": len(self.seen_hashes),
        }


class QualityPipeline:
    """Main quality pipeline orchestrator."""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        self.filter = DocumentFilter(config)
        self.deduplicator = Deduplicator(config)
        self.stats = {
            "input_documents": 0,
            "input_tokens": 0,
            "output_documents": 0,
            "output_tokens": 0,
            "filter_stats": {},
            "dedup_stats": {},
        }
    
    def process_document(self, doc: Document) -> Optional[Document]:
        """Process a single document through the pipeline."""
        self.stats["input_documents"] += 1
        self.stats["input_tokens"] += len(doc.text)
        
        # Apply filters
        if not self.filter.filter(doc):
            return None
        
        # Check duplicates
        if self.deduplicator.is_duplicate(doc):
            return None
        
        # Document passes
        self.stats["output_documents"] += 1
        self.stats["output_tokens"] += len(doc.text)
        return doc
    
    def process_stream(self, documents: Iterator[Document]) -> Iterator[Document]:
        """Process a stream of documents."""
        for doc in documents:
            result = self.process_document(doc)
            if result:
                yield result
    
    def get_stats(self) -> Dict:
        stats = self.stats.copy()
        stats["filter_stats"] = self.filter.get_stats()
        stats["dedup_stats"] = self.deduplicator.get_stats()
        stats["token_reduction"] = 1.0 - (stats["output_tokens"] / max(1, stats["input_tokens"]))
        return stats


def create_quality_config(profile: str = "default") -> QualityConfig:
    """Create quality configuration for different profiles."""
    
    configs = {
        "default": QualityConfig(),
        "strict": QualityConfig(
            min_doc_length=200,
            min_word_count=50,
            min_char_diversity=0.2,
            max_repetition_ratio=0.2,
            min_alphanumeric_ratio=0.6,
            remove_machine_generated=True,
        ),
        "permissive": QualityConfig(
            min_doc_length=50,
            min_word_count=10,
            min_char_diversity=0.1,
            max_repetition_ratio=0.4,
            min_alphanumeric_ratio=0.4,
            remove_machine_generated=False,
        ),
        "code": QualityConfig(
            min_doc_length=100,
            min_word_count=20,
            min_char_diversity=0.15,
            max_repetition_ratio=0.4,  # Code has repetition
            min_alphanumeric_ratio=0.4,  # Code has symbols
            remove_boilerplate=False,
            remove_navigation=False,
        ),
        "math": QualityConfig(
            min_doc_length=50,
            min_word_count=10,
            min_char_diversity=0.1,
            max_repetition_ratio=0.3,
            remove_boilerplate=False,
            remove_navigation=False,
        ),
    }
    
    return configs.get(profile, configs["default"])


def main():
    parser = argparse.ArgumentParser(description="Data Quality Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Input file/directory")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument("--config", type=str, default="default", 
                        choices=["default", "strict", "permissive", "code", "math"],
                        help="Quality profile")
    parser.add_argument("--stats-output", type=str, default=None, help="Stats output file")
    
    args = parser.parse_args()
    
    config = create_quality_config(args.config)
    pipeline = QualityPipeline(config)
    
    # This would process actual documents - placeholder for now
    print(f"Quality pipeline initialized with profile: {args.config}")
    print(f"Config: {asdict(config)}")
    
    # Save stats if requested
    if args.stats_output:
        with open(args.stats_output, "w") as f:
            json.dump(pipeline.get_stats(), f, indent=2)


if __name__ == "__main__":
    main()