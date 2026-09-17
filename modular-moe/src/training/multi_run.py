"""Multi-run training with best-run selection for Prothynesis."""

from __future__ import annotations

import os
import json
import time
import torch
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_trainer import BaseTrainer, TrainingState
from .checkpoint import CheckpointManager, CheckpointMetadata


@dataclass
class RunMetrics:
    """Metrics for a single training run."""
    run_id: int
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    reasoning_score: float = 0.0
    knowledge_score: float = 0.0
    domain_score: float = 0.0
    verification_score: float = 0.0
    stability_score: float = 0.0
    overfit_score: float = 0.0
    composite_score: float = 0.0
    checkpoint_path: str = ""
    duration_seconds: float = 0.0
    timestamp: str = ""
    tokens_seen: int = 0

    def compute_composite(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Compute composite quality score."""
        if weights is None:
            weights = {
                "val_loss": 0.3,
                "reasoning_score": 0.2,
                "knowledge_score": 0.15,
                "domain_score": 0.1,
                "verification_score": 0.1,
                "stability_score": 0.1,
                "overfit_score": 0.05,
            }

        # Lower is better for losses, higher is better for scores
        val_loss_component = 1.0 / (1.0 + self.val_loss) if self.val_loss != float("inf") else 0.0
        train_loss_component = 1.0 / (1.0 + self.train_loss) if self.train_loss != float("inf") else 0.0

        score = (
            weights.get("val_loss", 0.0) * val_loss_component +
            weights.get("train_loss", 0.0) * train_loss_component +
            weights.get("reasoning_score", 0.0) * self.reasoning_score +
            weights.get("knowledge_score", 0.0) * self.knowledge_score +
            weights.get("domain_score", 0.0) * self.domain_score +
            weights.get("verification_score", 0.0) * self.verification_score +
            weights.get("stability_score", 0.0) * self.stability_score +
            weights.get("overfit_score", 0.0) * self.overfit_score
        )

        self.composite_score = score
        return score

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MultiRunResult:
    """Result of multi-run training."""
    best_run_id: int
    best_run_metrics: RunMetrics
    all_runs: List[RunMetrics]
    total_runs: int
    total_duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "best_run_id": self.best_run_id,
            "best_run_metrics": self.best_run_metrics.to_dict(),
            "all_runs": [r.to_dict() for r in self.all_runs],
            "total_runs": self.total_runs,
            "total_duration_seconds": self.total_duration_seconds,
        }


class MultiRunTrainer:
    """
    Manages multiple training runs and selects the best one.
    
    Supports:
    - Multiple training runs with different seeds
    - Composite quality scoring
    - Best-run selection based on validation metrics
    - Automatic checkpoint reload for evaluation
    """

    def __init__(
        self,
        model_class: type,
        train_dataloader,
        eval_dataloader,
        config: Dict[str, Any],
        output_dir: str = "checkpoints/multi_run",
        num_runs: int = 3,
        score_weights: Optional[Dict[str, float]] = None,
        trainer_class: type = None,
    ):
        self.model_class = model_class
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_runs = num_runs
        self.score_weights = score_weights or {
            "val_loss": 0.3,
            "reasoning_score": 0.2,
            "knowledge_score": 0.15,
            "domain_score": 0.1,
            "verification_score": 0.1,
            "stability_score": 0.1,
            "overfit_score": 0.05,
        }
        self.runs: List[RunMetrics] = []
        self.trainer_class = trainer_class

    def run_all(self) -> MultiRunResult:
        """Execute all training runs and select the best."""
        start_time = time.time()

        for run_id in range(self.num_runs):
            print(f"\n{'='*60}")
            print(f"RUN {run_id + 1}/{self.num_runs}")
            print(f"{'='*60}")

            run_start = time.time()
            metrics = self._run_single(run_id)
            metrics.run_id = run_id
            metrics.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            metrics.duration_seconds = time.time() - run_start

            metrics.compute_composite(self.score_weights)
            self.runs.append(metrics)

            print(f"\nRun {run_id + 1} complete:")
            print(f"  Train loss: {metrics.train_loss:.4f}")
            print(f"  Val loss: {metrics.val_loss:.4f}")
            print(f"  Composite score: {metrics.composite_score:.4f}")
            print(f"  Duration: {metrics.duration_seconds:.1f}s")

        # Select best run
        best_run = max(self.runs, key=lambda r: r.composite_score)
        total_duration = time.time() - start_time

        result = MultiRunResult(
            best_run_id=best_run.run_id,
            best_run_metrics=best_run,
            all_runs=list(self.runs),
            total_runs=self.num_runs,
            total_duration_seconds=total_duration,
        )

        # Save results
        self._save_results(result)

        print(f"\n{'='*60}")
        print(f"BEST RUN: {best_run.run_id + 1}")
        print(f"  Composite score: {best_run.composite_score:.4f}")
        print(f"  Val loss: {best_run.val_loss:.4f}")
        print(f"  Checkpoint: {best_run.checkpoint_path}")
        print(f"{'='*60}")

        return result

    def _run_single(self, run_id: int) -> RunMetrics:
        """Execute a single training run."""
        # Set seed for reproducibility
        seed = self.config.get("training", {}).get("seed", 42) + run_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Create model
        model = self.model_class(**self._get_model_kwargs())

        # Create trainer
        run_dir = self.output_dir / f"run_{run_id:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        trainer_cls = self.trainer_class or BaseTrainer
        trainer = trainer_cls(
            model=model,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            config=self.config,
            output_dir=str(run_dir),
        )
        trainer.setup_optimizer()
        trainer.train()

        # Evaluate
        eval_metrics = trainer.evaluate()
        train_loss = eval_metrics.get("eval_loss", float("inf"))

        # Save checkpoint path
        checkpoint_path = str(run_dir / "final_model.safetensors")

        # Run additional evaluations
        reasoning_score = self._evaluate_reasoning(model)
        knowledge_score = self._evaluate_knowledge(model)
        domain_score = self._evaluate_domain(model)
        verification_score = self._evaluate_verification(model)
        stability_score = self._evaluate_stability(model)
        overfit_score = self._compute_overfit_score(train_loss, eval_metrics.get("eval_loss", train_loss))

        return RunMetrics(
            run_id=run_id,
            train_loss=train_loss,
            val_loss=eval_metrics.get("eval_loss", train_loss),
            reasoning_score=reasoning_score,
            knowledge_score=knowledge_score,
            domain_score=domain_score,
            verification_score=verification_score,
            stability_score=stability_score,
            overfit_score=overfit_score,
            checkpoint_path=checkpoint_path,
        )

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model constructor kwargs from config."""
        model_config = self.config.get("model", self.config.get("solver", {}))
        return model_config

    def _evaluate_reasoning(self, model: torch.nn.Module) -> float:
        """Evaluate reasoning capability (placeholder)."""
        return 0.5 + 0.1 * np.random.rand()

    def _evaluate_knowledge(self, model: torch.nn.Module) -> float:
        """Evaluate knowledge capability (placeholder)."""
        return 0.5 + 0.1 * np.random.rand()

    def _evaluate_domain(self, model: torch.nn.Module) -> float:
        """Evaluate domain coverage (placeholder)."""
        return 0.5 + 0.1 * np.random.rand()

    def _evaluate_verification(self, model: torch.nn.Module) -> float:
        """Evaluate verification capability (placeholder)."""
        return 0.5 + 0.1 * np.random.rand()

    def _evaluate_stability(self, model: torch.nn.Module) -> float:
        """Evaluate training stability (placeholder)."""
        return 0.5 + 0.1 * np.random.rand()

    def _compute_overfit_score(self, train_loss: float, val_loss: float) -> float:
        """Compute overfitting score (higher is better = less overfitting)."""
        if train_loss == float("inf") or val_loss == float("inf"):
            return 0.0
        gap = val_loss - train_loss
        # Positive gap = overfitting, negative = underfitting
        # Score: 1.0 when gap ~ 0, decreases as gap increases
        return max(0.0, 1.0 - gap * 10.0)

    def _save_results(self, result: MultiRunResult) -> None:
        """Save multi-run results."""
        results_path = self.output_dir / "multi_run_results.json"
        with open(results_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def load_best_run(self) -> str:
        """Load the best run checkpoint."""
        if not self.runs:
            raise RuntimeError("No runs completed")
        best_run = max(self.runs, key=lambda r: r.composite_score)
        return best_run.checkpoint_path


class RunSearchStrategy:
    """Strategies for multi-run search."""

    @staticmethod
    def grid_search(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate grid search configurations."""
        from itertools import product
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        configs = []
        for combo in product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        return configs

    @staticmethod
    def random_search(param_grid: Dict[str, List[Any]], num_trials: int = 10, seed: int = 42) -> List[Dict[str, Any]]:
        """Generate random search configurations."""
        rng = np.random.RandomState(seed)
        configs = []
        for _ in range(num_trials):
            config = {}
            for key, values in param_grid.items():
                config[key] = rng.choice(values)
            configs.append(config)
        return configs
