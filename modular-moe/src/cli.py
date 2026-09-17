"""CLI for Prothynesis."""

from __future__ import annotations

import sys
import os
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_config(path: str):
    from src.utils.config import load_config
    return load_config(path)


def cmd_model_list(args):
    """List registered models."""
    from momm.registry import create_registry
    registry = create_registry(registry_dir=args.registry_dir)
    models = registry.list_models()
    if not models:
        print("No models registered.")
        return 0
    print(f"{'Model ID':<20} {'Params':>10} {'Specialization':<20} {'Status'}")
    print("-" * 70)
    for m in models:
        print(f"{m.model_id:<20} {m.parameter_count:>10,} {m.specialization:<20} {m.training_status}")
    return 0


def cmd_model_inspect(args):
    """Inspect a model."""
    from momm.registry import create_registry
    registry = create_registry(registry_dir=args.registry_dir)
    metadata = registry.get_metadata(args.model_id)
    if not metadata:
        print(f"Model not found: {args.model_id}")
        return 1
    print(json.dumps(metadata.to_dict(), indent=2))
    return 0


def cmd_pool_generate(args):
    """Generate model pool."""
    from momm.registry import ModelRegistry, ModelMetadata
    from momm.model import ModelConfig, create_model
    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    count = config["solver"]["count"]
    registry_dir = args.output or config.get("training", {}).get("checkpoint_dir", "models")
    registry = ModelRegistry(registry_dir=registry_dir)

    print(f"Generating {count} model profiles...")

    for i in range(count):
        model_id = f"solver_{i:05d}"
        model_config = ModelConfig(
            model_id=model_id,
            vocab_size=config["solver"]["vocab_size"],
            max_seq_len=config["solver"]["max_seq_len"],
            hidden_dim=config["solver"]["hidden_dim"],
            num_layers=config["solver"]["num_layers"],
            num_heads=config["solver"]["num_heads"],
            head_dim=config["solver"]["head_dim"],
            ffn_hidden_dim=config["solver"]["ffn_hidden_dim"],
            activation=config["solver"]["activation"],
            norm_type=config["solver"]["norm_type"],
            norm_eps=1e-5,
            bias=False,
            dropout=config["solver"]["dropout"],
            attn_dropout=0.0,
            resid_dropout=0.0,
            use_flash_attn=False,
            tie_embeddings=config["solver"]["tie_embeddings"],
            use_rmsnorm=True,
            gradient_checkpointing=False,
            specialization="general",
            tokenizer_id="default",
        )

        model = create_model(model_config)
        metadata = ModelMetadata(
            model_id=model_id,
            checkpoint_path="",
            parameter_count=model.get_parameter_count()["total"],
            architecture=model_config.to_dict(),
            tokenizer_id="default",
            specialization="general",
            version="1.0",
            training_status="untrained",
        )
        registry.models[model_id] = metadata

    registry._save_manifest()
    print(f"Generated {count} model profiles in {registry_dir}")
    return 0


def cmd_train_single(args):
    """Train a single model (expert) from config."""
    import torch
    import numpy as np
    import tempfile
    from pathlib import Path

    from src.utils.config import load_config
    from src.utils.hardware import set_deterministic
    from src.model import create_model_from_config, create_expert
    from src.training.expert_trainer import ExpertTrainer
    from src.data.dataset import TokenizedDataset, create_dataloader
    from src.data.sharding import prepare_and_shard_dataset, get_shard_for_expert
    from src.data.tokenizer import CharacterTokenizer

    if not args.config:
        print("--config is required for train single")
        return 1

    config = load_config(args.config)
    set_deterministic(config.get("training", {}).get("seed", 42))

    expert_id = args.expert_id if hasattr(args, "expert_id") and args.expert_id is not None else 0
    max_steps = args.max_steps if hasattr(args, "max_steps") and args.max_steps is not None else config.get("training", {}).get("expert_pretrain", {}).get("max_steps", 10)

    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer = CharacterTokenizer(vocab_size=config["data"]["tokenizer_vocab_size"])
        metadata = prepare_and_shard_dataset(config, f"{tmpdir}/shards", tokenizer)
        shards = get_shard_for_expert(f"{tmpdir}/shards", expert_id, include_shared=True)
        combined = np.concatenate(shards)

        temp_path = Path(tmpdir) / f"expert_{expert_id}.bin"
        combined.astype(np.uint16).tofile(temp_path)

        try:
            dataset = TokenizedDataset(
                data_path=str(temp_path),
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

            expert = create_expert(
                hidden_dim=config["model"]["hidden_dim"],
                expert_hidden_dim=config["model"]["expert_hidden_dim"],
                expert_id=expert_id,
                config=config["model"],
            )

            output_dir = args.output or str(Path(tmpdir) / "checkpoints" / "experts")
            trainer = ExpertTrainer(
                expert=expert,
                expert_id=expert_id,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                config=config,
                output_dir=output_dir,
            )
            trainer.setup_optimizer()
            trainer.config["training"]["expert_pretrain"]["max_steps"] = max_steps
            trainer.train()
            expert_path = trainer.save_expert_only(trainer.state.step)
            print(f"Expert {expert_id} trained and saved to {expert_path}")
            return 0
        finally:
            if temp_path.exists():
                temp_path.unlink()


def cmd_train_pool(args):
    """Train model pool."""
    import subprocess
    import sys

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "run_quality_first_training.py"
    if not script_path.exists():
        print(f"Training script not found: {script_path}")
        return 1

    cmd = [
        sys.executable, str(script_path),
        "--config", args.config,
    ]
    if hasattr(args, "output_dir") and args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if hasattr(args, "num_models") and args.num_models:
        cmd.extend(["--num-models", str(args.num_models)])

    print(f"Running pool training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    return result.returncode


def cmd_train_orchestrator(args):
    """Train orchestrator."""
    import subprocess
    import sys

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "train_router.py"
    if not script_path.exists():
        print(f"Orchestrator training script not found: {script_path}")
        return 1

    cmd = [
        sys.executable, str(script_path),
        "--config", args.config,
    ]
    if hasattr(args, "output_dir") and args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    print(f"Running orchestrator training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    return result.returncode


def cmd_dataset_download(args):
    """Download datasets."""
    print(f"Downloading dataset: {args.dataset}")
    print("Dataset download requires network access and dataset libraries.")
    return 0


def cmd_dataset_prepare(args):
    """Prepare datasets."""
    import yaml
    from src.data.sharding import prepare_and_shard_dataset
    from src.data.tokenizer import create_tokenizer

    if not args.config:
        print("--config is required for dataset prepare")
        return 1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = args.output or config.get("data", {}).get("output_dir", "data/shards")
    tokenizer = create_tokenizer(config.get("data", {}))
    metadata = prepare_and_shard_dataset(config, output_dir, tokenizer)
    print(f"Prepared {len(metadata)} shards in {output_dir}")
    return 0


def cmd_dataset_shard(args):
    """Shard datasets."""
    import yaml
    from src.data.sharding import prepare_and_shard_dataset
    from src.data.tokenizer import create_tokenizer

    if not args.config:
        print("--config is required for dataset shard")
        return 1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    output_dir = args.output or config.get("data", {}).get("output_dir", "data/shards")
    tokenizer = create_tokenizer(config.get("data", {}))
    metadata = prepare_and_shard_dataset(config, output_dir, tokenizer)
    print(f"Sharded dataset into {len(metadata)} shards in {output_dir}")
    return 0


def cmd_dataset_stats(args):
    """Show dataset statistics."""
    import json
    import yaml
    from pathlib import Path

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    shard_dir = config.get("data", {}).get("output_dir", "data/shards")
    metadata_path = Path(shard_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"Metadata not found: {metadata_path}")
        return 1

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    total_tokens = sum(m["num_tokens"] for m in metadata)
    print(f"Shard directory : {shard_dir}")
    print(f"Total shards    : {len(metadata)}")
    print(f"Total tokens    : {total_tokens:,}")
    print(f"Expert shards   : {sum(1 for m in metadata if not m['is_shared'])}")
    print(f"Shared shards   : {sum(1 for m in metadata if m['is_shared'])}")
    return 0


def cmd_dataset_assign(args):
    """Assign unique dataset samples to models."""
    import yaml
    from src.data.assignment import create_assignment_engine, SampleRecord
    from src.data.sharding import prepare_and_shard_dataset
    from src.data.tokenizer import create_tokenizer
    from src.data.scheduler import create_scheduler

    if not args.config:
        print("--config is required for dataset assign")
        return 1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pool_size = config.get("solver", {}).get("count", 8)
    model_ids = [f"solver_{i:05d}" for i in range(pool_size)]

    # Load/prepare shards
    shard_dir = args.shard_dir or config.get("data", {}).get("output_dir", "data/shards")
    metadata_path = Path(shard_dir) / "metadata.json"
    if not metadata_path.exists():
        print(f"Shard metadata not found at {metadata_path}, preparing...")
        tokenizer = create_tokenizer(config.get("data", {}))
        prepare_and_shard_dataset(config, shard_dir, tokenizer)

    # Load tokenized data from shards
    import numpy as np
    shard_metadata = json.loads(metadata_path.read_text())
    all_tokens = []
    for meta in shard_metadata:
        path = Path(shard_dir) / meta["path"]
        if path.exists():
            data = np.fromfile(str(path), dtype=np.uint16)
            all_tokens.extend(data.tolist())

    # Create sample records from tokens
    from src.data.scheduler import DatasetProfile
    profiles = [
        DatasetProfile(
            name=f"shard_{m['shard_id']}",
            domain="general",
            difficulty="D2",
            task_type="reasoning",
            language="en",
            quality_score=1.0,
            size=m["num_tokens"],
            source="synthetic",
            license="open",
        )
        for m in shard_metadata
    ]

    engine = create_assignment_engine(config.get("datasets", {}))
    samples = []
    for i, token in enumerate(all_tokens):
        samples.append(
            SampleRecord(
                sample_id=f"tok_{i:08d}",
                text=str(token),
                token_count=1,
                domain="general",
                difficulty="D2",
                task_type="reasoning",
                language="en",
                quality_score=1.0,
                source="synthetic",
            )
        )
    engine.load_samples(samples)

    # Build shared core
    shared = engine.build_shared_core()
    print(f"Shared core: {len(shared)} samples")

    # Assign unique experience
    scheduler = create_scheduler(config)
    assignments = engine.assign_unique_experience(model_ids, scheduler, seed=42)

    # Save assignment
    output_path = args.output or str(Path(shard_dir) / "assignment.json")
    engine.save_assignment(output_path)
    print(f"Assignment saved to {output_path}")

    # Report
    report = engine.check_ownership()
    print(f"Total samples       : {report.total_samples:,}")
    print(f"Shared core         : {report.shared_core_count:,}")
    print(f"Unique pool         : {report.unique_pool_count:,}")
    print(f"Assigned            : {report.assigned_count:,}")
    print(f"Unassigned          : {report.unassigned_count:,}")
    print(f"Duplicates          : {report.duplicate_count}")
    print(f"Duplicate %         : {report.duplicate_percentage:.2f}%")
    return 0


def cmd_dataset_ownership_check(args):
    """Check dataset ownership and duplication."""
    import yaml
    from src.data.assignment import create_assignment_engine
    from pathlib import Path

    if not args.config:
        print("--config is required for dataset ownership-check")
        return 1

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    assignment_path = args.assignment_path or str(
        Path(config.get("data", {}).get("output_dir", "data/shards")) / "assignment.json"
    )

    if not Path(assignment_path).exists():
        print(f"Assignment file not found: {assignment_path}")
        return 1

    engine = create_assignment_engine(config.get("datasets", {}))
    engine.load_assignment(assignment_path)

    report = engine.check_ownership()
    print(json.dumps(report.to_dict(), indent=2))

    # Diversity check
    pool_size = config.get("solver", {}).get("count", 8)
    model_ids = [f"solver_{i:05d}" for i in range(pool_size)]
    diversity = engine.check_population_diversity(model_ids)
    print("\nPopulation Diversity:")
    print(json.dumps(diversity, indent=2))

    if diversity.get("population_diversity_warning"):
        print("\nWARNING: Population diversity is too low!")
        return 1

    return 0


def cmd_memory_inspect(args):
    """Inspect model memory."""
    print(f"Inspecting memory for model: {args.model_id}")
    print("Memory inspection requires an active task context.")
    return 0


def cmd_evaluate(args):
    """Evaluate models."""
    import torch
    import yaml
    from pathlib import Path

    from src.utils.config import load_config
    from src.model import create_model_from_config
    from src.evaluation import evaluate_perplexity, evaluate_routing_detailed
    from src.data.dataset import TokenizedDataset, create_dataloader

    config_path = args.config if hasattr(args, "config") and args.config else None
    if not config_path:
        print("--config is required for evaluate")
        return 1

    config = load_config(config_path)
    model_path = args.model_path if hasattr(args, "model_path") and args.model_path else None
    if not model_path:
        print("--model-path is required for evaluate")
        return 1

    device = torch.device("cpu")
    model = create_model_from_config(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    tokenizer_path = config.get("data", {}).get("tokenizer_path")
    if not tokenizer_path:
        print("tokenizer_path missing in config data section")
        return 1

    dataset = TokenizedDataset(
        data_path=tokenizer_path,
        tokenizer=None,
        max_seq_len=config["model"]["max_seq_len"],
    )
    eval_dataset = torch.utils.data.Subset(dataset, range(min(20, len(dataset))))
    eval_dataloader = create_dataloader(eval_dataset, batch_size=4, num_workers=0)

    ppl_results = evaluate_perplexity(model, eval_dataloader, device, max_batches=10)
    print(f"Perplexity: {ppl_results.get('perplexity', 'N/A'):.2f}")

    routing_results = evaluate_routing_detailed(model, eval_dataloader, device, max_batches=10)
    print(f"Dead experts: {routing_results.get('global_dead_experts', 'N/A')}")
    print(f"Load balance CV: {routing_results.get('global_load_balance_cv', 'N/A'):.4f}")
    return 0


def cmd_simulate(args):
    """Run simulation."""
    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    pool_size = config.get("solver", {}).get("count", 0)
    print(f"Simulating with {pool_size} models...")
    print("Use the GUI Orchestration tab for interactive simulation.")
    return 0


def cmd_benchmark(args):
    """Run benchmarks."""
    print(f"Running benchmarks with config: {args.config}")
    print("Use the GUI Benchmarks tab for interactive benchmarking.")
    return 0


def cmd_chat(args):
    """Start chat."""
    print("Starting chat...")
    print("Use the GUI Chat tab for interactive chat.")
    return 0


def cmd_package_build(args):
    """Build PMO package."""
    from pmo.package import create_pmo_package
    import yaml

    if not args.source:
        print("--source is required for package build")
        return 1

    config_path = args.config if hasattr(args, "config") and args.config else None
    if not config_path:
        print("--config is required for package build")
        return 1

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    package = create_pmo_package(args.source, config)
    print(f"Built PMO package at {args.source}")
    return 0


def cmd_package_inspect(args):
    """Inspect PMO package."""
    from pmo.package import PMOPackage

    package = PMOPackage(args.package)
    info = package.inspect()
    print(json.dumps(info, indent=2))
    return 0


def cmd_package_verify(args):
    """Verify PMO package."""
    from pmo.package import PMOPackage

    package = PMOPackage(args.package)
    valid = package.verify()
    print(f"PMO package {'valid' if valid else 'invalid'}: {args.package}")
    return 0 if valid else 1


def build_parser():
    """Build argument parser."""
    parser = argparse.ArgumentParser(
        prog="prothynesis",
        description="Prothynesis - Pure Hierarchical Orchestral MoMMs",
    )
    parser.add_argument("--registry-dir", default="models", help="Model registry directory")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # model
    model_parser = subparsers.add_parser("model", help="Model management")
    model_sub = model_parser.add_subparsers(dest="model_command")

    list_parser = model_sub.add_parser("list", help="List models")
    list_parser.set_defaults(func=cmd_model_list)

    inspect_parser = model_sub.add_parser("inspect", help="Inspect model")
    inspect_parser.add_argument("model_id", help="Model ID to inspect")
    inspect_parser.set_defaults(func=cmd_model_inspect)

    # pool
    pool_parser = subparsers.add_parser("pool", help="Pool management")
    pool_sub = pool_parser.add_subparsers(dest="pool_command")

    generate_parser = pool_sub.add_parser("generate", help="Generate model pool")
    generate_parser.add_argument("--config", required=True, help="Config YAML path")
    generate_parser.add_argument("--output", help="Output registry directory")
    generate_parser.set_defaults(func=cmd_pool_generate)

    # train
    train_parser = subparsers.add_parser("train", help="Training")
    train_sub = train_parser.add_subparsers(dest="train_command")

    single_parser = train_sub.add_parser("single", help="Train single model")
    single_parser.add_argument("model_id", help="Model ID")
    single_parser.add_argument("--config", required=True, help="Config YAML")
    single_parser.add_argument("--expert-id", type=int, default=0, help="Expert ID for expert pretraining")
    single_parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    single_parser.add_argument("--output", default=None, help="Output directory")
    single_parser.set_defaults(func=cmd_train_single)

    pool_train_parser = train_sub.add_parser("pool", help="Train pool")
    pool_train_parser.add_argument("--config", required=True, help="Config YAML")
    pool_train_parser.add_argument("--output-dir", default=None, help="Output directory")
    pool_train_parser.add_argument("--num-models", type=int, default=None, help="Number of models")
    pool_train_parser.set_defaults(func=cmd_train_pool)

    orch_parser = train_sub.add_parser("orchestrator", help="Train orchestrator")
    orch_parser.add_argument("--config", required=True, help="Config YAML")
    orch_parser.add_argument("--output-dir", default=None, help="Output directory")
    orch_parser.set_defaults(func=cmd_train_orchestrator)

    # dataset
    dataset_parser = subparsers.add_parser("dataset", help="Dataset management")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command")

    download_parser = dataset_sub.add_parser("download", help="Download dataset")
    download_parser.add_argument("dataset", help="Dataset name")
    download_parser.set_defaults(func=cmd_dataset_download)

    prepare_parser = dataset_sub.add_parser("prepare", help="Prepare datasets")
    prepare_parser.add_argument("--config", required=True, help="Config YAML")
    prepare_parser.add_argument("--output", default=None, help="Output directory")
    prepare_parser.set_defaults(func=cmd_dataset_prepare)

    shard_parser = dataset_sub.add_parser("shard", help="Shard datasets")
    shard_parser.add_argument("--config", required=True, help="Config YAML")
    shard_parser.add_argument("--output", default=None, help="Output directory")
    shard_parser.set_defaults(func=cmd_dataset_shard)

    stats_parser = dataset_sub.add_parser("stats", help="Dataset statistics")
    stats_parser.add_argument("--config", required=True, help="Config YAML")
    stats_parser.set_defaults(func=cmd_dataset_stats)

    assign_parser = dataset_sub.add_parser("assign", help="Assign unique dataset samples to models")
    assign_parser.add_argument("--config", required=True, help="Config YAML")
    assign_parser.add_argument("--shard-dir", default=None, help="Shard directory")
    assign_parser.add_argument("--output", default=None, help="Output assignment path")
    assign_parser.set_defaults(func=cmd_dataset_assign)

    ownership_parser = dataset_sub.add_parser("ownership-check", help="Check dataset ownership and duplication")
    ownership_parser.add_argument("--config", required=True, help="Config YAML")
    ownership_parser.add_argument("--assignment-path", default=None, help="Assignment JSON path")
    ownership_parser.set_defaults(func=cmd_dataset_ownership_check)

    # memory
    memory_parser = subparsers.add_parser("memory", help="Memory inspection")
    memory_sub = memory_parser.add_subparsers(dest="memory_command")

    inspect_mem_parser = memory_sub.add_parser("inspect", help="Inspect model memory")
    inspect_mem_parser.add_argument("model_id", help="Model ID")
    inspect_mem_parser.set_defaults(func=cmd_memory_inspect)

    # evaluate
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--config", required=True, help="Config YAML")
    eval_parser.add_argument("--model-path", required=True, help="Model checkpoint path")
    eval_parser.set_defaults(func=cmd_evaluate)

    # simulate
    sim_parser = subparsers.add_parser("simulate", help="Run simulation")
    sim_parser.add_argument("--config", required=True, help="Config YAML")
    sim_parser.set_defaults(func=cmd_simulate)

    # benchmark
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmarks")
    bench_parser.add_argument("--config", required=True, help="Config YAML")
    bench_parser.set_defaults(func=cmd_benchmark)

    # chat
    chat_parser = subparsers.add_parser("chat", help="Start chat")
    chat_parser.set_defaults(func=cmd_chat)

    # package
    package_parser = subparsers.add_parser("package", help="PMO package management")
    package_sub = package_parser.add_subparsers(dest="package_command")

    build_parser = package_sub.add_parser("build", help="Build PMO package")
    build_parser.add_argument("--source", required=True, help="Source directory")
    build_parser.add_argument("--config", required=True, help="Config YAML")
    build_parser.set_defaults(func=cmd_package_build)

    inspect_pkg_parser = package_sub.add_parser("inspect", help="Inspect PMO package")
    inspect_pkg_parser.add_argument("package", help="PMO file path")
    inspect_pkg_parser.set_defaults(func=cmd_package_inspect)

    verify_parser = package_sub.add_parser("verify", help="Verify PMO package")
    verify_parser.add_argument("package", help="PMO file path")
    verify_parser.set_defaults(func=cmd_package_verify)

    return parser


def main():
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        return args.func(args)
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
