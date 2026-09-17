"""Tiny end-to-end test of the full pipeline."""

import sys
import os
import shutil
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

from src.utils.config import load_config
from src.utils.hardware import set_deterministic
from src.utils.paths import get_temp_root, is_allowed_dataset_path
from src.model import create_model_from_config, MoELanguageModel
from src.training.expert_trainer import ExpertTrainer
from src.training.router_trainer import RouterTrainer
from src.training.joint_trainer import JointTrainer
from src.data.dataset import TokenizedDataset, create_dataloader
from src.data.tokenizer import CharacterTokenizer
from src.data.sharding import prepare_and_shard_dataset, verify_shards


def _make_tmpdir(prefix: str = "tiny_e2e_") -> Path:
    base = get_temp_root() / prefix
    base.mkdir(parents=True, exist_ok=True)
    return base


def _remove_tmpdir(path: Path) -> None:
    if path.exists() and is_allowed_dataset_path(path):
        shutil.rmtree(path, ignore_errors=True)


def test_tiny_pipeline():
    """Test the complete tiny pipeline."""
    # Use tiny config
    config = load_config("configs/tiny_test.yaml")
    
    # Override for faster testing
    config["training"]["expert_pretrain"]["max_steps"] = 2
    config["training"]["router_train"]["max_steps"] = 1
    config["training"]["joint_train"]["max_steps"] = 2
    config["training"]["expert_pretrain"]["batch_size"] = 1
    config["training"]["router_train"]["batch_size"] = 1
    config["training"]["joint_train"]["batch_size"] = 1
    
    tmpdir = _make_tmpdir()
    try:
        # ===== Stage A: Dataset Preparation =====
        print("\n=== Stage A: Dataset Preparation ===")
        tokenizer = CharacterTokenizer(vocab_size=config["data"]["tokenizer_vocab_size"])
        
        metadata = prepare_and_shard_dataset(config, f"{tmpdir}/shards", tokenizer)
        assert len(metadata) == 2  # 2 experts
        
        # Verify
        assert verify_shards(f"{tmpdir}/shards")
        print("[OK] Dataset prepared and sharded")
        
        # ===== Stage B: Expert Pretraining =====
        print("\n=== Stage B: Expert Pretraining ===")
        expert_checkpoints = []
        
        for expert_id in range(2):
            print(f"Training expert {expert_id}...")
            
            # Load expert shard
            from src.data.sharding import get_shard_for_expert
            shards = get_shard_for_expert(f"{tmpdir}/shards", expert_id, include_shared=True)
            combined = np.concatenate(shards)
            
            # Save to temp file under approved temp root
            temp_path = str(get_temp_root() / f"tiny_e2e_expert_{expert_id}_{int(time.time())}.bin")
            with open(temp_path, "wb") as f:
                f.write(combined.astype(np.uint16).tobytes())
            
            try:
                # Dataset
                dataset = TokenizedDataset(
                    data_path=temp_path,
                    tokenizer=tokenizer,
                    max_seq_len=config["model"]["max_seq_len"],
                )
                
                train_dataloader = create_dataloader(
                    dataset,
                    batch_size=config["training"]["expert_pretrain"]["batch_size"],
                    num_workers=0,
                    shuffle=True,
                )
                
                eval_dataset = torch.utils.data.Subset(dataset, range(min(10, len(dataset))))
                eval_dataloader = create_dataloader(eval_dataset, batch_size=4, num_workers=0)
                
                # Create expert
                from src.model import create_expert
                expert = create_expert(
                    hidden_dim=config["model"]["hidden_dim"],
                    expert_hidden_dim=config["model"]["expert_hidden_dim"],
                    expert_id=expert_id,
                    config=config["model"],
                )
                
                # Trainer
                trainer = ExpertTrainer(
                    expert=expert,
                    expert_id=expert_id,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    config=config,
                    output_dir=f"{tmpdir}/checkpoints/experts",
                )
                trainer.setup_optimizer()
                trainer.train()
                
                # Save expert weights
                expert_path = trainer.save_expert_only(trainer.state.step)
                expert_checkpoints.append(expert_path)
                print(f"  [OK] Expert {expert_id} trained and saved")
                
            finally:
                os.unlink(temp_path)
        
        # ===== Stage D: MoE Assembly =====
        print("\n=== Stage D: MoE Assembly ===")
        model = create_model_from_config(config)
        
        # Load expert weights
        for expert_id, ckpt_path in enumerate(expert_checkpoints):
            model.load_expert_checkpoint(expert_id, ckpt_path)
        
        print("[OK] MoE assembled with pretrained experts")
        
        # ===== Stage E: Router Training =====
        print("\n=== Stage E: Router Training ===")
        
        # Combine all shards for router training
        all_shards = []
        for meta in metadata:
            path = Path(f"{tmpdir}/shards") / meta.path
            all_shards.append(np.fromfile(path, dtype=np.uint16))
        all_tokens = np.concatenate(all_shards)
        
        temp_path = str(get_temp_root() / f"tiny_e2e_router_{int(time.time())}.bin")
        with open(temp_path, "wb") as f:
            f.write(all_tokens.astype(np.uint16).tobytes())
        
        try:
            router_dataset = TokenizedDataset(
                data_path=temp_path,
                tokenizer=tokenizer,
                max_seq_len=config["model"]["max_seq_len"],
            )
            
            router_train_dataloader = create_dataloader(
                router_dataset,
                batch_size=config["training"]["router_train"]["batch_size"],
                num_workers=0,
                shuffle=True,
            )
            
            router_eval_dataloader = create_dataloader(
                torch.utils.data.Subset(router_dataset, range(min(10, len(router_dataset)))),
                batch_size=4, num_workers=0,
            )
            
            router_trainer = RouterTrainer(
                model=model,
                train_dataloader=router_train_dataloader,
                eval_dataloader=router_eval_dataloader,
                config=config,
                output_dir=f"{tmpdir}/checkpoints/router",
            )
            router_trainer.setup_optimizer()
            router_trainer.train()
            print("[OK] Router trained")
            
        finally:
            os.unlink(temp_path)
        
        # ===== Stage F: Joint Training =====
        print("\n=== Stage F: Joint Training ===")
        
        joint_trainer = JointTrainer(
            model=model,
            train_dataloader=router_train_dataloader,
            eval_dataloader=router_eval_dataloader,
            config=config,
            output_dir=f"{tmpdir}/checkpoints/global",
        )
        joint_trainer.setup_optimizer()
        joint_trainer.train()
        print("[OK] Joint training complete")
        
        # ===== Stage G: Evaluation =====
        print("\n=== Stage G: Evaluation ===")
        from src.evaluation import evaluate_perplexity, evaluate_routing_detailed
        
        eval_dataloader = create_dataloader(
            torch.utils.data.Subset(router_dataset, range(min(5, len(router_dataset)))),
            batch_size=4, num_workers=0,
        )
        
        device = torch.device("cpu")
        model.to(device)
        
        ppl_results = evaluate_perplexity(model, eval_dataloader, device, max_batches=5)
        print(f"  Perplexity: {ppl_results.get('perplexity', 'N/A'):.2f}")
        
        routing_results = evaluate_routing_detailed(model, eval_dataloader, device, max_batches=5)
        print(f"  Dead experts: {routing_results.get('global_dead_experts', 'N/A')}")
        print(f"  Load balance CV: {routing_results.get('global_load_balance_cv', 'N/A'):.4f}")
        
        print("\n[OK] Full pipeline test passed!")
    finally:
        _remove_tmpdir(tmpdir)


if __name__ == "__main__":
    test_tiny_pipeline()