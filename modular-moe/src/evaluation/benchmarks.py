"""Benchmark system for Prothynesis."""

from __future__ import annotations

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    score: float
    latency_ms: float
    models_recruited: int
    parameter_evaluations: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "latency_ms": self.latency_ms,
            "models_recruited": self.models_recruited,
            "parameter_evaluations": self.parameter_evaluations,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class BenchmarkSuite:
    """A collection of benchmarks."""
    name: str
    benchmarks: List[str]
    results: List[BenchmarkResult] = field(default_factory=list)
    
    def add_result(self, result: BenchmarkResult) -> None:
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of benchmark results."""
        if not self.results:
            return {"status": "no_results"}
        
        scores = [r.score for r in self.results]
        latencies = [r.latency_ms for r in self.results]
        
        return {
            "total_benchmarks": len(self.results),
            "average_score": round(sum(scores) / len(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "average_latency_ms": round(sum(latencies) / len(latencies), 1),
            "total_models_recruited": sum(r.models_recruited for r in self.results),
            "total_parameter_evaluations": sum(r.parameter_evaluations for r in self.results),
        }


class BenchmarkRunner:
    """Run benchmarks on Prothynesis models."""
    
    BENCHMARK_CONFIGS = {
        "mmlu": {"domains": 57, "questions": 14042},
        "math": {"domains": 7, "questions": 12500},
        "coding": {"domains": 8, "questions": 5000},
        "reasoning": {"domains": 10, "questions": 3000},
    }
    
    def __init__(self, config: dict):
        self.config = config
        self.results: List[BenchmarkResult] = []
    
    def run_benchmark(self, name: str, models_count: int = 10) -> BenchmarkResult:
        """Run a single benchmark."""
        bench_config = self.BENCHMARK_CONFIGS.get(name, {"questions": 1000})
        
        start_time = time.time()
        
        # Simulate running benchmark
        base_score = 0.5
        score = min(0.99, base_score + (models_count * 0.02) + random.uniform(-0.05, 0.05))
        latency = random.uniform(50, 500)
        models_used = max(1, models_count // 3)
        param_evals = models_used * bench_config["questions"] * 75_000_000
        
        result = BenchmarkResult(
            name=name,
            score=round(score, 4),
            latency_ms=round(latency, 1),
            models_recruited=models_used,
            parameter_evaluations=param_evals,
            metadata={
                "questions": bench_config["questions"],
                "domains": bench_config.get("domains", 1),
            }
        )
        
        self.results.append(result)
        return result
    
    def run_suite(self, benchmark_names: List[str], models_count: int = 10) -> BenchmarkSuite:
        """Run a suite of benchmarks."""
        suite = BenchmarkSuite(name="default", benchmarks=benchmark_names)
        
        for name in benchmark_names:
            result = self.run_benchmark(name, models_count)
            suite.add_result(result)
        
        return suite
    
    def compare_modes(self, benchmark_names: List[str]) -> Dict[str, BenchmarkSuite]:
        """Compare single model vs flat vs hierarchical."""
        modes = {
            "single_75m": 1,
            "flat_momm": 10,
            "hierarchical_momm": 10,
        }
        
        results = {}
        for mode, model_count in modes.items():
            suite = self.run_suite(benchmark_names, models_count=model_count)
            results[mode] = suite
        
        return results


def create_benchmark_runner(config: dict) -> BenchmarkRunner:
    """Factory function to create benchmark runner."""
    return BenchmarkRunner(config)
