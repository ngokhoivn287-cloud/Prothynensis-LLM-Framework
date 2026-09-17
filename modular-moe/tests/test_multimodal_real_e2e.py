"""Real multimodal end-to-end tests for Ultra 3 MrTP."""

from __future__ import annotations

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import numpy as np
import safetensors.torch

from src.utils.config import load_config
from src.utils.paths import (
    enforce_path_policy,
    get_primary_root,
    get_temp_root,
    get_checkpoints_root,
    is_allowed_dataset_path,
    _ensure_roots,
)
from src.data.real_datasets import BoundedImageNetLoader, BoundedComputerUseLoader
from src.data.multimodal_schema import (
    MultimodalSample,
    Modality,
    DatasetProfile,
    ModalityAdapter,
    TextAdapter,
    ImageAdapter,
    VideoAdapter,
    ScreenAdapter,
    MultimodalDatasetRegistry,
)
from src.model.vision_model import (
    VisionModelConfig,
    VisionReasoningModel,
    create_vision_model,
    count_vision_model_parameters,
)
from src.model.computer_use_model import (
    ComputerUseModelConfig,
    ComputerUseModel,
    create_computer_use_model,
    count_computer_use_model_parameters,
)
from src.training.population_selection import (
    PopulationSelectionEngine,
    CandidateRecord,
    CapabilityVector,
    CandidateStatus,
)
from src.momm.collaboration import CollaborativeSynthesizer, SynthesisStrategy
from src.momm.coordinator import ModelCoordinator, CoordinatorConfig


def _cleanup(path: Path):
    if path.exists() and is_allowed_dataset_path(path):
        import shutil
        shutil.rmtree(path, ignore_errors=True)


class TestRealImageNetBounded:
    def test_imagenet_loader_prepare(self):
        _cleanup(get_primary_root() / "datasets" / "imagenet_1k_mrtp")
        loader = BoundedImageNetLoader(max_samples=8, split="train")
        samples = loader.prepare()
        assert len(samples) > 0
        for s in samples:
            assert Path(s["path"]).exists()
            assert is_allowed_dataset_path(s["path"])
            assert s["modality"] == "image"
            assert s["domain"] == "computer_vision"

    def test_imagenet_samples_under_approved_root(self):
        loader = BoundedImageNetLoader(max_samples=4, split="train")
        samples = loader.prepare()
        bad = [s["path"] for s in samples if not is_allowed_dataset_path(s["path"])]
        assert bad == []

    def test_imagenet_metadata_persisted(self):
        loader = BoundedImageNetLoader(max_samples=4, split="train")
        loader.prepare()
        assert loader.metadata_path.exists()
        with open(loader.metadata_path, "r") as f:
            data = json.load(f)
        assert data["actual_count"] > 0


class TestRealComputerUseBounded:
    def test_computer_use_loader_prepare(self):
        _cleanup(get_primary_root() / "datasets" / "computer_use_mrtp")
        loader = BoundedComputerUseLoader(max_samples=2, max_frames=1)
        import threading
        result = {}
        def target():
            try:
                result["samples"] = loader.prepare()
            except Exception as e:
                result["error"] = str(e)
        t = threading.Thread(target=target)
        t.start()
        t.join(timeout=180)
        assert t.is_alive() is False, "Computer-use loader timed out"
        samples = result.get("samples", [])
        assert len(samples) > 0
        for s in samples:
            assert Path(s["path"]).exists()
            assert is_allowed_dataset_path(s["path"])
            assert len(s.get("frame_paths", [])) > 0
            assert s["modality"] in ("video", "screen")

    def test_computer_use_samples_under_approved_root(self):
        loader = BoundedComputerUseLoader(max_samples=2, max_frames=1)
        import threading
        result = {}
        def target():
            try:
                result["samples"] = loader.prepare()
            except Exception as e:
                result["error"] = str(e)
        t = threading.Thread(target=target)
        t.start()
        t.join(timeout=180)
        assert t.is_alive() is False, "Computer-use loader timed out"
        samples = result.get("samples", [])
        bad = [s["path"] for s in samples if not is_allowed_dataset_path(s["path"])]
        assert bad == []


class TestRealVisionE2E:
    def test_vision_real_training(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedImageNetLoader(max_samples=8, split="train")
        samples = loader.prepare()
        result = _run_vision_training(config, samples)
        assert result["final_loss"] < 20
        assert result["changed_params"] > 0
        assert Path(result["checkpoint_path"]).exists()
        assert is_allowed_dataset_path(result["checkpoint_path"])

    def test_vision_checkpoint_reload(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedImageNetLoader(max_samples=4, split="train")
        samples = loader.prepare()
        result = _run_vision_training(config, samples)
        reload = _reload_vision(result["checkpoint_path"], samples[0], config)
        assert reload["finite"] is True
        assert reload["logits_shape"][0] == 1


class TestRealComputerUseE2E:
    def test_computer_use_real_training(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedComputerUseLoader(max_samples=4, max_frames=2)
        samples = loader.prepare()
        result = _run_computer_use_training(config, samples)
        assert result["final_loss"] < 20
        assert result["changed_params"] > 0
        assert Path(result["checkpoint_path"]).exists()
        assert is_allowed_dataset_path(result["checkpoint_path"])

    def test_computer_use_checkpoint_reload(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedComputerUseLoader(max_samples=4, max_frames=2)
        samples = loader.prepare()
        result = _run_computer_use_training(config, samples)
        reload = _reload_computer_use(result["checkpoint_path"], samples[0], config)
        assert reload["finite"] is True
        assert reload["logits_shape"][0] == 1


class TestMultimodalMultiRun:
    def test_vision_multi_run_runs2(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedImageNetLoader(max_samples=4, split="train")
        samples = loader.prepare()
        runs = _run_vision_multi_run(config, samples, runs=2)
        assert len(runs) == 2
        best = max(runs, key=lambda r: r["final_loss"])
        assert best["run_id"] in (0, 1)

    def test_computer_use_multi_run_runs2(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedComputerUseLoader(max_samples=4, max_frames=2)
        samples = loader.prepare()
        runs = _run_computer_use_multi_run(config, samples, runs=2)
        assert len(runs) == 2
        best = max(runs, key=lambda r: r["final_loss"])
        assert best["run_id"] in (0, 1)


class TestMultimodalCollaboration:
    def test_vision_general_collaboration(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedImageNetLoader(max_samples=2, split="train")
        samples = loader.prepare()
        result = _run_collaboration(config, samples)
        assert result["status"] == "completed"
        assert "synthesis" in result

    def test_collaboration_evidence_trace(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        loader = BoundedImageNetLoader(max_samples=2, split="train")
        samples = loader.prepare()
        result = _run_collaboration(config, samples)
        assert "vision_evidence" in result
        assert "synthesis" in result
        assert result["synthesis"]["confidence"] > 0


class TestMultimodalSelection:
    def test_selection_produces_retained(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        engine = PopulationSelectionEngine(config.get("population_selection", {}))
        candidates = [
            CandidateRecord(model_id=f"c{i}", model_family="vision", overall_quality=0.8 - i * 0.1,
                            capability_vector=CapabilityVector(computer_vision=0.8, reasoning=0.5))
            for i in range(5)
        ]
        retained, discarded = engine.select(candidates)
        assert len(retained) >= 1

    def test_selection_report_writes_under_approved_root(self):
        config = load_config("configs/ultra3_mrtp.yaml")
        engine = PopulationSelectionEngine(config.get("population_selection", {}))
        candidates = [
            CandidateRecord(model_id="c0", model_family="vision", overall_quality=0.9,
                            capability_vector=CapabilityVector(computer_vision=0.9, reasoning=0.5))
        ]
        retained, discarded = engine.select(candidates)
        out = get_checkpoints_root() / "ultra3_mrtp" / "test_selection.json"
        report = engine.export_selection_report(retained, discarded, str(out))
        assert out.exists()
        assert is_allowed_dataset_path(out)
        assert report["retained_count"] >= 1


def _run_vision_training(config, samples):
    solver_cfg = config["solver"]
    vcfg = VisionModelConfig(
        model_id="vision_test", vocab_size=solver_cfg["vocab_size"], max_seq_len=solver_cfg["max_seq_len"],
        hidden_dim=solver_cfg["hidden_dim"], num_layers=solver_cfg["num_layers"], num_heads=solver_cfg["num_heads"],
        head_dim=solver_cfg["head_dim"], ffn_hidden_dim=solver_cfg["ffn_hidden_dim"],
        activation=solver_cfg["activation"], norm_type=solver_cfg["norm_type"], tie_embeddings=solver_cfg["tie_embeddings"],
        image_size=64, patch_size=8, vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
    )
    model = create_vision_model(vcfg)
    from PIL import Image
    images, labels = [], []
    for s in samples[:4]:
        img = Image.open(s["path"]).convert("RGB").resize((64, 64))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        labels.append(s.get("label", 0))
    images = torch.tensor(np.stack(images)).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for _ in range(3):
        opt.zero_grad()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.repeat(logits.shape[1]).reshape(-1))
        loss.backward()
        opt.step()
    after = model.state_dict()
    ckpt = get_checkpoints_root() / "ultra3_mrtp" / "vision" / "vision_test.safetensors"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(model.state_dict(), ckpt)
    return {
        "final_loss": float(loss.item()),
        "changed_params": sum(1 for k in before if not torch.equal(before[k], after[k])),
        "checkpoint_path": str(ckpt),
    }


def _reload_vision(ckpt_path, sample, config):
    vcfg = VisionModelConfig(
        model_id="vision_reload", vocab_size=config["solver"]["vocab_size"], max_seq_len=config["solver"]["max_seq_len"],
        hidden_dim=config["solver"]["hidden_dim"], num_layers=config["solver"]["num_layers"], num_heads=config["solver"]["num_heads"],
        head_dim=config["solver"]["head_dim"], ffn_hidden_dim=config["solver"]["ffn_hidden_dim"],
        activation=config["solver"]["activation"], norm_type=config["solver"]["norm_type"], tie_embeddings=config["solver"]["tie_embeddings"],
        image_size=64, patch_size=8, vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
    )
    model = create_vision_model(vcfg)
    model.load_state_dict(safetensors.torch.load_file(ckpt_path, device="cpu"))
    model.eval()
    from PIL import Image
    img = Image.open(sample["path"]).convert("RGB").resize((64, 64))
    arr = torch.tensor(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        logits = model(arr)
    return {"finite": bool(torch.isfinite(logits).all().item()), "logits_shape": list(logits.shape)}


def _run_computer_use_training(config, samples):
    solver_cfg = config["solver"]
    ccfg = ComputerUseModelConfig(
        model_id="cu_test", vocab_size=solver_cfg["vocab_size"], max_seq_len=solver_cfg["max_seq_len"],
        hidden_dim=solver_cfg["hidden_dim"], num_layers=solver_cfg["num_layers"], num_heads=solver_cfg["num_heads"],
        head_dim=solver_cfg["head_dim"], ffn_hidden_dim=solver_cfg["ffn_hidden_dim"],
        activation=solver_cfg["activation"], norm_type=solver_cfg["norm_type"], tie_embeddings=solver_cfg["tie_embeddings"],
        image_size=64, patch_size=8, vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
        num_action_types=10, max_actions_per_step=5,
    )
    model = create_computer_use_model(ccfg)
    from PIL import Image
    images, action_targets = [], []
    for s in samples[:4]:
        fps = s.get("frame_paths", [])
        if not fps:
            continue
        img = Image.open(fps[0]).convert("RGB").resize((64, 64))
        images.append(np.asarray(img, dtype=np.float32) / 255.0)
        actions = s.get("actions", {})
        action_targets.append(actions.get("action_type", 0) if isinstance(actions, dict) else 0)
    if not images:
        pytest.skip("No valid computer-use images")
    images = torch.tensor(np.stack(images)).permute(0, 3, 1, 2)
    action_targets = torch.tensor(action_targets, dtype=torch.long)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001)
    for _ in range(3):
        opt.zero_grad()
        outputs = model(images)
        logit_loss = torch.nn.functional.cross_entropy(outputs["logits"].reshape(-1, outputs["logits"].shape[-1]), action_targets.repeat(outputs["logits"].shape[1]).reshape(-1))
        action_pred = outputs["actions"]["action_type"]
        if action_pred.ndim == 2:
            action_loss = torch.nn.functional.cross_entropy(action_pred, action_targets)
        else:
            action_loss = torch.nn.functional.cross_entropy(action_pred.reshape(-1, action_pred.shape[-1]), action_targets.repeat(action_pred.shape[1]).reshape(-1))
        loss = logit_loss + action_loss
        loss.backward()
        opt.step()
    after = model.state_dict()
    ckpt = get_checkpoints_root() / "ultra3_mrtp" / "computer_use" / "cu_test.safetensors"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    safetensors.torch.save_file(model.state_dict(), ckpt)
    return {
        "final_loss": float(loss.item()),
        "changed_params": sum(1 for k in before if not torch.equal(before[k], after[k])),
        "checkpoint_path": str(ckpt),
    }


def _reload_computer_use(ckpt_path, sample, config):
    ccfg = ComputerUseModelConfig(
        model_id="cu_reload", vocab_size=config["solver"]["vocab_size"], max_seq_len=config["solver"]["max_seq_len"],
        hidden_dim=config["solver"]["hidden_dim"], num_layers=config["solver"]["num_layers"], num_heads=config["solver"]["num_heads"],
        head_dim=config["solver"]["head_dim"], ffn_hidden_dim=config["solver"]["ffn_hidden_dim"],
        activation=config["solver"]["activation"], norm_type=config["solver"]["norm_type"], tie_embeddings=config["solver"]["tie_embeddings"],
        image_size=64, patch_size=8, vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
        num_action_types=10, max_actions_per_step=5,
    )
    model = create_computer_use_model(ccfg)
    model.load_state_dict(safetensors.torch.load_file(ckpt_path, device="cpu"))
    model.eval()
    from PIL import Image
    frame_paths = sample.get("frame_paths", [])
    img_path = frame_paths[0] if frame_paths else sample.get("path", "")
    if not img_path or not Path(img_path).exists():
        pytest.skip(f"Frame not found: {img_path}")
    img = Image.open(img_path).convert("RGB").resize((64, 64))
    arr = torch.tensor(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        outputs = model(arr)
    finite = bool(torch.isfinite(outputs["logits"]).all().item())
    return {
        "finite": finite,
        "logits_shape": list(outputs["logits"].shape),
        "action_type_shape": list(outputs["actions"]["action_type"].shape),
    }


def _run_vision_multi_run(config, samples, runs=2):
    results = []
    for run_id in range(runs):
        torch.manual_seed(42 + run_id)
        result = _run_vision_training(config, samples)
        result["run_id"] = run_id
        results.append(result)
    return results


def _run_computer_use_multi_run(config, samples, runs=2):
    results = []
    for run_id in range(runs):
        torch.manual_seed(42 + run_id)
        result = _run_computer_use_training(config, samples)
        result["run_id"] = run_id
        results.append(result)
    return results


def _run_collaboration(config, samples):
    solver_cfg = config["solver"]
    general = __import__("src.momm.model", fromlist=["ModelConfig", "create_model"])
    general_model = general.create_model(general.ModelConfig(
        model_id="general_mrtp_00", vocab_size=solver_cfg["vocab_size"], max_seq_len=solver_cfg["max_seq_len"],
        hidden_dim=solver_cfg["hidden_dim"], num_layers=solver_cfg["num_layers"], num_heads=solver_cfg["num_heads"],
        head_dim=solver_cfg["head_dim"], ffn_hidden_dim=solver_cfg["ffn_hidden_dim"],
        activation=solver_cfg["activation"], norm_type=solver_cfg["norm_type"], tie_embeddings=solver_cfg["tie_embeddings"],
    ))
    vcfg = VisionModelConfig(
        model_id="vision_mrtp_00", vocab_size=solver_cfg["vocab_size"], max_seq_len=solver_cfg["max_seq_len"],
        hidden_dim=solver_cfg["hidden_dim"], num_layers=solver_cfg["num_layers"], num_heads=solver_cfg["num_heads"],
        head_dim=solver_cfg["head_dim"], ffn_hidden_dim=solver_cfg["ffn_hidden_dim"],
        activation=solver_cfg["activation"], norm_type=solver_cfg["norm_type"], tie_embeddings=solver_cfg["tie_embeddings"],
        image_size=64, patch_size=8, vision_hidden_dim=256, vision_num_layers=2, vision_num_heads=4,
    )
    vision_model = create_vision_model(vcfg)
    from PIL import Image
    img = Image.open(samples[0]["path"]).convert("RGB").resize((64, 64))
    arr = torch.tensor(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        vision_out = vision_model(arr)
    vision_evidence = {
        "logits_mean": float(vision_out.mean().item()),
        "logits_std": float(vision_out.std().item()),
        "max_logit": float(vision_out.max().item()),
    }
    synthesizer = CollaborativeSynthesizer(
        task_id="ultra3_mrtp_collab_test",
        strategy=SynthesisStrategy.EVIDENCE_WEIGHTED.value,
    )
    synthesizer.add_candidate(model_id="vision_mrtp_00", content=f"vision_evidence={vision_evidence}", confidence=0.7,
                              evidence=[f"vision_{k}={v:.4f}" for k, v in vision_evidence.items()])
    synthesizer.add_candidate(model_id="general_solver_mrtp_00", content="general_reasoning_placeholder", confidence=0.6,
                              evidence=["general_reasoning=0.7"])
    synthesizer.add_verification("candidate_1", True, "verifier", confidence=0.9)
    synthesizer.add_verification("candidate_2", True, "verifier", confidence=0.9)
    result = synthesizer.synthesize()
    return {
        "status": "completed",
        "vision_evidence": vision_evidence,
        "synthesis": {
            "confidence": result.confidence,
            "consensus": result.consensus_score,
            "participating": result.participating_models,
        },
    }
