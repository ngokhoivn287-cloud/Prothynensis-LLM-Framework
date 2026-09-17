"""Multimodal population engine for Vision and Computer-Use models."""

from __future__ import annotations

import os
import sys
import json
import time
import tempfile
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Optional, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from src.utils.config import load_config
from src.utils.hardware import set_deterministic
from src.utils.paths import get_temp_root, get_checkpoints_root, is_allowed_dataset_path
from src.data.tokenizer import CharacterTokenizer
from src.data.sharding import prepare_and_shard_dataset, get_shard_for_expert
from src.data.dataset import TokenizedDataset, create_dataloader
from src.data.scheduler import DatasetScheduler, DatasetProfile
from src.data.multimodal_schema import (
    MultimodalSample,
    Modality,
    ModalityAdapter,
    MultimodalDatasetRegistry,
    TextAdapter,
    ImageAdapter,
    VideoAdapter,
    ScreenAdapter,
)
from src.momm.model import ModelConfig, create_model
from src.momm.registry import ModelRegistry
from src.model.vision_model import VisionModelConfig, VisionReasoningModel, create_vision_model
from src.model.computer_use_model import (
    ComputerUseModelConfig,
    ComputerUseModel,
    create_computer_use_model,
)
from src.training.expert_trainer import ExpertTrainer
from src.training.population_selection import (
    PopulationSelectionEngine,
    CandidateRecord,
    CandidateStatus,
    CapabilityVector,
)
from src.training.adaptive_budget import AdaptiveBudgetAllocator
from src.training.population_scheduler import PopulationScheduler, ModelTrainingRecord, ModelTrainingStatus
from src.training.quality_lock import QualityLock
from src.training.lineage import LineageTracker
from src.training.evolution import PopulationEvolution, GenerationManifest
from src.training.checkpoint import CheckpointManager


logger = logging.getLogger(__name__)


class MultimodalPopulationEngine:
    """Trains and selects Vision and Computer-Use model populations."""

    def __init__(self, config: Dict[str, Any], tmpdir: Optional[str] = None):
        self.config = config
        self.tmpdir = tmpdir or str(get_temp_root() / f"multimodal_{int(time.time())}")
        Path(self.tmpdir).mkdir(parents=True, exist_ok=True)
        self.registry = MultimodalDatasetRegistry()
        self.selection_engine = PopulationSelectionEngine(config.get("population_selection", {}))
        self.lineage = LineageTracker(config)
        self.evolution = PopulationEvolution(config)
        self._register_default_profiles()

    def _register_default_profiles(self) -> None:
        self.registry.register_profile(DatasetProfile(
            name="imagenet_1k",
            domain="computer_vision",
            difficulty="D3",
            task_type="image_classification",
            modality=Modality.IMAGE,
            quality_score=0.95,
            size=1281167,
            source="ILSVRC/imagenet-1k",
            license="custom",
        ))
        self.registry.register_profile(DatasetProfile(
            name="computer_use_large",
            domain="computer_use",
            difficulty="D4",
            task_type="gui_interaction",
            modality=Modality.VIDEO,
            quality_score=0.90,
            size=1000000,
            source="markov-ai/computer-use-large",
            license="custom",
            metadata={"streaming": True, "partial_download": True, "resumable": True},
        ))

    def prepare_image_sample(self, image_path: str, domain: str = "computer_vision") -> Optional[MultimodalSample]:
        adapter = ImageAdapter()
        sample = MultimodalSample(
            sample_id=f"img_{Path(image_path).stem}",
            modality=Modality.IMAGE,
            domain=domain,
            task="image_understanding",
            difficulty="D2",
            quality=0.85,
            payload_path=image_path,
        )
        return sample

    def prepare_screen_sample(self, image_path: str, domain: str = "computer_use") -> Optional[MultimodalSample]:
        sample = MultimodalSample(
            sample_id=f"screen_{Path(image_path).stem}",
            modality=Modality.SCREEN,
            domain=domain,
            task="gui_grounding",
            difficulty="D3",
            quality=0.85,
            payload_path=image_path,
            annotations={"screen_state": "unknown"},
        )
        return sample

    def preprocess_sample(self, sample: MultimodalSample) -> Optional[Dict[str, Any]]:
        return self.registry.preprocess_sample(sample)

    def register_vision_model(self, model_id: str, checkpoint_path: str = "") -> VisionReasoningModel:
        solver_cfg = self.config["solver"]
        cfg = VisionModelConfig(
            model_id=model_id,
            vocab_size=solver_cfg["vocab_size"],
            max_seq_len=solver_cfg["max_seq_len"],
            hidden_dim=solver_cfg["hidden_dim"],
            num_layers=solver_cfg["num_layers"],
            num_heads=solver_cfg["num_heads"],
            head_dim=solver_cfg["head_dim"],
            ffn_hidden_dim=solver_cfg["ffn_hidden_dim"],
            activation=solver_cfg["activation"],
            norm_type=solver_cfg["norm_type"],
            tie_embeddings=solver_cfg["tie_embeddings"],
        )
        model = create_vision_model(cfg)
        return model

    def register_computer_use_model(self, model_id: str, checkpoint_path: str = "") -> ComputerUseModel:
        solver_cfg = self.config["solver"]
        cfg = ComputerUseModelConfig(
            model_id=model_id,
            vocab_size=solver_cfg["vocab_size"],
            max_seq_len=solver_cfg["max_seq_len"],
            hidden_dim=solver_cfg["hidden_dim"],
            num_layers=solver_cfg["num_layers"],
            num_heads=solver_cfg["num_heads"],
            head_dim=solver_cfg["head_dim"],
            ffn_hidden_dim=solver_cfg["ffn_hidden_dim"],
            activation=solver_cfg["activation"],
            norm_type=solver_cfg["norm_type"],
            tie_embeddings=solver_cfg["tie_embeddings"],
        )
        model = create_computer_use_model(cfg)
        return model

    def train_vision_candidate(self, model_id: str, num_steps: int = 10) -> Dict[str, Any]:
        vision_cfg = self.config.get("vision_population", {})
        model = self.register_vision_model(model_id)
        dummy_image = torch.randn(1, 3, self.config["solver"]["image_size"], self.config["solver"]["image_size"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        model.train()
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            logits = model(dummy_image)
            loss = logits.mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        ckpt_dir = get_checkpoints_root() / "multimodal" / model_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / f"{model_id}_final.safetensors")
        torch.save(model.state_dict(), ckpt_path)
        return {"model_id": model_id, "final_loss": losses[-1], "checkpoint_path": ckpt_path, "losses": losses}

    def train_computer_use_candidate(self, model_id: str, num_steps: int = 10) -> Dict[str, Any]:
        model = self.register_computer_use_model(model_id)
        dummy_image = torch.randn(1, 3, self.config["solver"]["image_size"], self.config["solver"]["image_size"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        model.train()
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            outputs = model(dummy_image)
            loss = outputs["logits"].mean() + outputs["actions"]["action_type"].mean()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        ckpt_dir = get_checkpoints_root() / "multimodal" / model_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = str(ckpt_dir / f"{model_id}_final.safetensors")
        torch.save(model.state_dict(), ckpt_path)
        return {"model_id": model_id, "final_loss": losses[-1], "checkpoint_path": ckpt_path, "losses": losses}

    def build_candidate_record(self, model_id: str, model_family: str, training_result: Dict[str, Any],
                               capabilities: Optional[Dict[str, float]] = None) -> CandidateRecord:
        if capabilities is None:
            capabilities = {f: 0.5 + 0.3 * np.random.rand() for f in CapabilityVector.__dataclass_fields__}
        cap = CapabilityVector.from_dict(capabilities)
        record = CandidateRecord(
            model_id=model_id,
            model_family=model_family,
            capability_vector=cap,
            overall_quality=max(0.0, min(1.0, 1.0 / (1.0 + training_result.get("final_loss", 1.0)))),
            checkpoint_path=training_result.get("checkpoint_path", ""),
            training_run=training_result,
        )
        return record

    def select_population(self, candidates: List[CandidateRecord], target_count: Optional[int] = None,
                          model_family: Optional[str] = None) -> Dict[str, Any]:
        retained, discarded = self.selection_engine.select(candidates, target_count=target_count,
                                                           model_family=model_family)
        report_path = str(get_checkpoints_root() / "multimodal" / f"selection_{model_family or 'all'}.json")
        report = self.selection_engine.export_selection_report(retained, discarded, report_path)
        return {
            "retained": retained,
            "discarded": discarded,
            "report": report,
            "retained_count": len(retained),
            "discarded_count": len(discarded),
        }

    def build_hierarchy(self, retained: List[CandidateRecord],
                        hierarchy_spec: Optional[Dict[str, int]] = None) -> Dict[str, List[CandidateRecord]]:
        if hierarchy_spec is None:
            hierarchy_spec = {"ultimate": 1, "master": 1, "chief": 1, "orchestral": 1}
        return self.selection_engine.build_hierarchy(retained, hierarchy_spec)

    def run_multimodal_pipeline(self) -> Dict[str, Any]:
        vision_count = self.config.get("vision_population", {}).get("count", 0)
        cu_count = self.config.get("computer_use_population", {}).get("count", 0)
        vision_candidates = []
        cu_candidates = []
        for i in range(vision_count):
            model_id = f"vision_{i:03d}"
            result = self.train_vision_candidate(model_id, num_steps=5)
            record = self.build_candidate_record(model_id, "vision", result)
            vision_candidates.append(record)
        for i in range(cu_count):
            model_id = f"computer_use_{i:03d}"
            result = self.train_computer_use_candidate(model_id, num_steps=5)
            record = self.build_candidate_record(model_id, "computer_use", result)
            cu_candidates.append(record)
        vision_selection = self.select_population(vision_candidates, model_family="vision")
        cu_selection = self.select_population(cu_candidates, model_family="computer_use")
        all_retained = vision_selection["retained"] + cu_selection["retained"]
        hierarchy = self.build_hierarchy(all_retained)
        return {
            "vision_candidates": len(vision_candidates),
            "computer_use_candidates": len(cu_candidates),
            "vision_retained": len(vision_selection["retained"]),
            "computer_use_retained": len(cu_selection["retained"]),
            "vision_discarded": len(vision_selection["discarded"]),
            "computer_use_discarded": len(cu_selection["discarded"]),
            "hierarchy": {tier: [c.model_id for c in models] for tier, models in hierarchy.items()},
        }
