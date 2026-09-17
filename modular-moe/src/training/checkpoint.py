"""Checkpoint utilities for saving and loading model states."""

from __future__ import annotations

import os
import json
import torch
import safetensors.torch
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime

from ..utils.logging import get_logger
from ..utils.hardware import get_git_revision

try:
    from ..utils.paths import is_allowed_dataset_path, get_checkpoints_root, assert_allowed_path
except ImportError:
    def is_allowed_dataset_path(path):
        return True
    def get_checkpoints_root():
        return Path("checkpoints")
    def assert_allowed_path(path, context=""):
        return Path(path)

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for a checkpoint."""
    step: int
    epoch: int
    tokens_seen: int
    dataset_position: int
    config: Dict[str, Any]
    git_revision: Optional[str]
    timestamp: str
    metrics: Dict[str, float]
    rng_state: Dict[str, Any]


def has_shared_tensors(state_dict: Dict[str, torch.Tensor]) -> bool:
    """Check if state_dict has shared tensors (same memory)."""
    # Get data pointers for all tensors
    ptrs = {}
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        ptr = tensor.data_ptr()
        if ptr in ptrs:
            return True
        ptrs[ptr] = name
    return False


def atomic_save(obj: Any, path: Path) -> None:
    """Atomically save an object to disk."""
    # Save to temporary file first
    temp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, temp_path)
    # Atomic replace (works on Windows)
    os.replace(temp_path, path)


def atomic_save_safetensors(tensors: Dict[str, torch.Tensor], path: Path) -> None:
    """Atomically save tensors using safetensors."""
    temp_path = path.with_suffix(path.suffix + ".tmp")
    safetensors.torch.save_file(tensors, temp_path)
    os.replace(temp_path, path)


class CheckpointManager:
    """Manages model checkpoints with atomic writes and rotation."""
    
    def __init__(
        self,
        root_dir: str,
        format: str = "safetensors",
        keep_last_n: int = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        save_rng: bool = True,
    ):
        root_path = Path(root_dir)
        if not is_allowed_dataset_path(root_path):
            root_path = get_checkpoints_root() / Path(root_dir).name
        self.root_dir = root_path
        self.format = format
        self.keep_last_n = keep_last_n
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.save_rng = save_rng
        
        self.root_dir.mkdir(parents=True, exist_ok=True)
    
    def save(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        step: int = 0,
        epoch: int = 0,
        tokens_seen: int = 0,
        dataset_position: int = 0,
        config: Optional[Dict] = None,
        metrics: Optional[Dict[str, float]] = None,
        rng_states: Optional[Dict] = None,
        prefix: str = "step",
    ) -> Path:
        """Save a checkpoint."""
        step_dir = self.root_dir / f"{prefix}_{step:08d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Model weights
        state_dict = model.state_dict()
        
        # Check for shared tensors - safetensors doesn't support shared tensors
        use_safetensors = self.format == "safetensors" and not has_shared_tensors(state_dict)
        
        if use_safetensors:
            model_path = step_dir / "model.safetensors"
            atomic_save_safetensors(state_dict, model_path)
        else:
            if self.format == "safetensors":
                logger.warning("Model has shared tensors, falling back to PyTorch format for checkpoint")
            model_path = step_dir / "model.pt"
            atomic_save(state_dict, model_path)
        
        # Optimizer state
        if self.save_optimizer and optimizer is not None:
            opt_path = step_dir / "optimizer.pt"
            atomic_save(optimizer.state_dict(), opt_path)
        
        # Scheduler state
        if self.save_scheduler and scheduler is not None:
            sched_path = step_dir / "scheduler.pt"
            atomic_save(scheduler.state_dict(), sched_path)
        
        # RNG states
        if self.save_rng:
            rng_path = step_dir / "rng.pt"
            rng_state = rng_states or self._get_rng_state()
            atomic_save(rng_state, rng_path)
        
        # Metadata
        metadata = CheckpointMetadata(
            step=step,
            epoch=epoch,
            tokens_seen=tokens_seen,
            dataset_position=dataset_position,
            config=config or {},
            git_revision=get_git_revision(),
            timestamp=datetime.now().isoformat(),
            metrics=metrics or {},
            rng_state={},  # Saved separately
        )
        
        meta_path = step_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)
        
        # Rotate old checkpoints
        self._rotate_checkpoints(prefix)
        
        logger.info(f"Saved checkpoint: {step_dir}")
        return step_dir
    
    def _get_rng_state(self) -> Dict[str, Any]:
        """Get all RNG states."""
        state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": None,
            "python": None,
        }
        
        try:
            import numpy as np
            state["numpy"] = np.random.get_state()
        except ImportError:
            pass
        
        try:
            import random
            state["python"] = random.getstate()
        except ImportError:
            pass
        
        return state
    
    def _set_rng_state(self, state: Dict[str, Any]) -> None:
        """Restore all RNG states."""
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
        
        if state["numpy"] is not None:
            try:
                import numpy as np
                np.random.set_state(state["numpy"])
            except Exception:
                pass
        
        if state["python"] is not None:
            try:
                import random
                random.setstate(state["python"])
            except Exception:
                pass
    
    def _rotate_checkpoints(self, prefix: str) -> None:
        """Remove old checkpoints beyond keep_last_n."""
        checkpoints = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith(f"{prefix}_")
        ])
        
        while len(checkpoints) > self.keep_last_n:
            old = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old)
            logger.debug(f"Removed old checkpoint: {old}")
    
    def load(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        load_rng: bool = True,
        strict: bool = True,
    ) -> CheckpointMetadata:
        """Load a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load model - check both formats
        safetensors_path = checkpoint_path / "model.safetensors"
        pytorch_path = checkpoint_path / "model.pt"
        
        if safetensors_path.exists():
            model_path = safetensors_path
            state_dict = safetensors.torch.load_file(model_path)
        elif pytorch_path.exists():
            model_path = pytorch_path
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
        else:
            raise FileNotFoundError(f"No model checkpoint found in {checkpoint_path}")
        
        model.load_state_dict(state_dict, strict=strict)
        
        # Load optimizer
        if self.save_optimizer and optimizer is not None:
            opt_path = checkpoint_path / "optimizer.pt"
            if opt_path.exists():
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu", weights_only=False))
        
        # Load scheduler
        if self.save_scheduler and scheduler is not None:
            sched_path = checkpoint_path / "scheduler.pt"
            if sched_path.exists():
                scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=False))
        
        # Load RNG
        if self.save_rng and load_rng:
            rng_path = checkpoint_path / "rng.pt"
            if rng_path.exists():
                rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
                self._set_rng_state(rng_state)
        
        # Load metadata
        meta_path = checkpoint_path / "metadata.json"
        with open(meta_path, "r") as f:
            metadata_dict = json.load(f)
        
        metadata = CheckpointMetadata(**metadata_dict)
        
        logger.info(f"Loaded checkpoint: {checkpoint_path} (step {metadata.step})")
        return metadata
    
    def find_latest(self, prefix: str = "step") -> Optional[Path]:
        """Find the latest checkpoint directory."""
        checkpoints = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith(f"{prefix}_")
        ])
        
        if checkpoints:
            return checkpoints[-1]
        return None
    
    def list_checkpoints(self, prefix: str = "step") -> List[Path]:
        """List all checkpoints sorted by step."""
        return sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.startswith(f"{prefix}_")
        ])
    
    def verify_checkpoint(self, checkpoint_path: Path) -> bool:
        """Verify a checkpoint is valid."""
        checkpoint_path = Path(checkpoint_path)
        
        # Check required files
        required = ["metadata.json"]
        if self.format == "safetensors":
            required.append("model.safetensors")
        else:
            required.append("model.pt")
        
        for req in required:
            if not (checkpoint_path / req).exists():
                logger.error(f"Missing required file: {req}")
                return False
        
        # Try loading metadata
        try:
            with open(checkpoint_path / "metadata.json", "r") as f:
                json.load(f)
        except Exception as e:
            logger.error(f"Invalid metadata: {e}")
            return False
        
        return True


def save_expert_checkpoint(
    expert: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    step: int,
    tokens_seen: int,
    dataset_position: int,
    config: Dict,
    metrics: Dict[str, float],
    output_dir: str,
    expert_id: int,
    format: str = "safetensors",
    keep_last_n: int = 3,
) -> Path:
    """Save a checkpoint for a single expert."""
    manager = CheckpointManager(
        root_dir=os.path.join(output_dir, f"expert_{expert_id:03d}"),
        format=format,
        keep_last_n=keep_last_n,
    )
    
    return manager.save(
        model=expert,
        optimizer=optimizer,
        scheduler=scheduler,
        step=step,
        tokens_seen=tokens_seen,
        dataset_position=dataset_position,
        config=config,
        metrics=metrics,
        prefix="step",
    )


def load_expert_checkpoint(
    expert: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    checkpoint_dir: str,
    load_rng: bool = True,
) -> CheckpointMetadata:
    """Load a checkpoint for a single expert."""
    manager = CheckpointManager(root_dir=checkpoint_dir)
    latest = manager.find_latest()
    
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    
    return manager.load(
        checkpoint_path=latest,
        model=expert,
        optimizer=optimizer,
        scheduler=scheduler,
        load_rng=load_rng,
    )