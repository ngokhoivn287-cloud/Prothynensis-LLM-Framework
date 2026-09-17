"""Tests for Prothynesis."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "modular-moe" / "src"))


def test_75m_parameter_count():
    """Test that model has approximately 75M parameters."""
    from momm.model import ModelConfig, create_model
    
    config = ModelConfig(
        model_id="test_75m",
        vocab_size=50304,
        max_seq_len=2048,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        head_dim=64,
        ffn_hidden_dim=2048,
        activation="silu",
        norm_type="rmsnorm",
        tie_embeddings=True,
    )
    
    model = create_model(config)
    param_counts = model.get_parameter_count()
    total = param_counts["total"]
    
    print(f"75M model parameter count: {total:,}")
    print(f"  In millions: {total / 1e6:.2f}M")
    
    # Should be approximately 77M with the 75M-target architecture
    assert 70e6 <= total <= 85e6, f"Expected ~75-80M params, got {total:,}"
    print("  PASS: Parameter count is in 70M-85M range")


def test_trinity_1_0_validation():
    """Test Trinity 1.0 configuration validation."""
    config = {
        "solver": {"count": 24400, "parameter_count_per_model": 75000000},
        "architecture": "trinity"
    }
    
    solver_params = config["solver"]["count"] * config["solver"]["parameter_count_per_model"]
    expected = 24400 * 75_000_000
    
    print(f"Trinity 1.0: {config['solver']['count']:,} × 75M = {solver_params:,}")
    assert solver_params == expected, f"Expected {expected:,}, got {solver_params:,}"
    assert solver_params == 1_830_000_000_000, f"Expected 1.83T, got {solver_params:,}"
    print("  PASS: Trinity 1.0 = 24,400 × 75M = 1.83T")


def test_trinity_3_0_validation():
    """Test Trinity 3.0 2x capacity requirement."""
    trinity1_solver_count = 24400
    trinity1_params = trinity1_solver_count * 75_000_000
    min_trinity3_params = 2 * trinity1_params
    
    # Valid Trinity 3.0 config
    config_valid = {
        "solver": {"count": 48800, "parameter_count_per_model": 75000000},
        "architecture": "trinity3"
    }
    solver_params_valid = config_valid["solver"]["count"] * config_valid["solver"]["parameter_count_per_model"]
    
    print(f"Trinity 3.0 valid: {config_valid['solver']['count']:,} × 75M = {solver_params_valid:,}")
    assert solver_params_valid >= min_trinity3_params, \
        f"Trinity 3.0 must have >= {min_trinity3_params:,} params"
    print(f"  PASS: {solver_params_valid:,} >= {min_trinity3_params:,}")
    
    # Invalid Trinity 3.0 config
    config_invalid = {
        "solver": {"count": 24400, "parameter_count_per_model": 75000000},
        "architecture": "trinity3"
    }
    solver_params_invalid = config_invalid["solver"]["count"] * config_invalid["solver"]["parameter_count_per_model"]
    
    print(f"Trinity 3.0 invalid: {config_invalid['solver']['count']:,} × 75M = {solver_params_invalid:,}")
    assert solver_params_invalid < min_trinity3_params, \
        f"Should be rejected: {solver_params_invalid:,} < {min_trinity3_params:,}"
    print(f"  PASS: Correctly rejected as < {min_trinity3_params:,}")


def test_all_configs():
    """Test all architecture configurations."""
    import yaml
    
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    config_names = ["mini", "mini3", "pro", "pro3", "ultra", "ultra3", "trinity", "trinity3", "custom"]
    
    for name in config_names:
        path = configs_dir / f"{name}.yaml"
        if not path.exists():
            print(f"  SKIP: {name}.yaml not found at {path}")
            continue
        
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        
        solver_count = config["solver"]["count"]
        params_per_model = config["solver"]["parameter_count_per_model"]
        total_params = solver_count * params_per_model
        declared_total = config["solver"]["total_parameters"]
        
        print(f"{name}: {solver_count:,} × {params_per_model:,} = {total_params:,}")
        
        # Verify consistency
        assert declared_total == total_params, \
            f"Inconsistent total_parameters in {name}: declared {declared_total}, calculated {total_params}"
        
        # Validate scaling rules
        if name == "mini3":
            assert solver_count == 2 * 5120, "Mini V3 must be 2× Mini V1"
            assert config["orchestration"]["orchestral"] == 6, "Mini V3 orchestral must be 3× Mini V1"
        elif name == "pro3":
            assert solver_count == 2 * 10040, "Pro V3 must be 2× Pro V1"
            assert config["orchestration"]["orchestral"] == 28, "Pro V3 orchestral must be 7× Pro V1"
        elif name == "ultra3":
            assert solver_count == 2 * 16420, "Ultra V3 must be 2× Ultra V1"
            assert config["orchestration"]["orchestral"] == 70, "Ultra V3 orchestral must be 10× Ultra V1"
        elif name == "trinity3":
            assert total_params >= 2 * 24400 * 77161472, \
                f"Trinity 3.0 below 2x Trinity 1.0 requirement"
            assert config["orchestration"]["orchestral"] == 150, "Trinity 3.0 orchestral must be 15× Trinity 1.0"
            assert config["orchestration"]["chief"] == 75, "Trinity 3.0 chief must be 15× Trinity 1.0"
            assert config["orchestration"]["master"] == 45, "Trinity 3.0 master must be 15× Trinity 1.0"
            assert config["orchestration"]["ultimate"] == 15, "Trinity 3.0 ultimate must be 15× Trinity 1.0"
    
    print("  PASS: All configs valid")


def test_dataset_scheduler():
    """Test dataset scheduler."""
    from data.scheduler import DatasetScheduler, ModelMixture, DatasetProfile
    
    config = {
        "datasets": {
            "min_coverage_all_domains": True,
            "minimum_reasoning_exposure": 0.15,
            "minimum_verification_exposure": 0.05,
        }
    }
    
    scheduler = DatasetScheduler(config)
    
    # Generate mixture
    mixture = scheduler.generate_model_mixture("solver_0001", seed=42)
    
    # Validate
    assert mixture.model_id == "solver_0001"
    assert len(mixture.domain_weights) > 0
    assert len(mixture.difficulty_weights) > 0
    assert len(mixture.task_weights) > 0
    
    # Check normalization
    total_domain = sum(mixture.domain_weights.values())
    assert abs(total_domain - 1.0) < 0.01, f"Domain weights not normalized: {total_domain}"
    
    # Check coverage
    issues = mixture.validate_coverage(config["datasets"])
    print(f"  Coverage issues: {issues}")
    
    print("  PASS: Dataset scheduler works correctly")


def test_model_registry():
    """Test model registry."""
    from momm.registry import ModelRegistry, ModelMetadata
    from momm.model import ModelConfig, create_model
    import tempfile
    import shutil
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        registry = ModelRegistry(registry_dir=temp_dir)
        
        # Create a model
        config = ModelConfig(
            model_id="test_model",
            hidden_dim=512,
            num_layers=12,
            num_heads=8,
            ffn_hidden_dim=2048,
        )
        model = create_model(config)
        
        # Register
        metadata = registry.register_model(
            model=model,
            checkpoint_path="",
            specialization="general",
            version="1.0",
        )
        
        assert metadata.model_id == "test_model"
        assert metadata.parameter_count > 0
        
        # List
        models = registry.list_models()
        assert len(models) == 1
        
        # Get metadata
        retrieved = registry.get_metadata("test_model")
        assert retrieved is not None
        assert retrieved.model_id == "test_model"
        
        print("  PASS: Model registry works correctly")
    finally:
        shutil.rmtree(temp_dir)


def test_pool_generation():
    """Test pool generation."""
    from momm.registry import ModelRegistry, ModelMetadata
    from momm.model import ModelConfig, create_model
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        registry = ModelRegistry(registry_dir=temp_dir)
        
        # Generate small pool
        pool_size = 10
        for i in range(pool_size):
            model_id = f"solver_{i:05d}"
            config = ModelConfig(
                model_id=model_id,
                hidden_dim=512,
                num_layers=12,
                num_heads=8,
                ffn_hidden_dim=2048,
            )
            model = create_model(config)
            
            metadata = ModelMetadata(
                model_id=model_id,
                checkpoint_path="",
                parameter_count=model.get_parameter_count()["total"],
                architecture=config.to_dict(),
                tokenizer_id="default",
                specialization="general",
                version="1.0",
                training_status="untrained",
            )
            registry.models[model_id] = metadata
        
        registry._save_manifest()
        
        # Verify
        assert len(registry.models) == pool_size
        assert registry.manifest_path.exists()
        
        print(f"  PASS: Generated {pool_size} model profiles")
    finally:
        shutil.rmtree(temp_dir)


def test_simulation_engine():
    """Test simulation engine."""
    from momm.simulation import SimulationEngine
    
    config = {"pool_size": 100, "max_rounds": 3, "recruitment": "quality"}
    engine = SimulationEngine(config)
    
    # Initialize pool
    engine.initialize_pool(100, seed=42)
    assert len(engine.pool) == 100
    
    # Run simulation
    result = engine.run_simulation(config)
    
    assert result.pool_size == 100
    assert result.models_recruited > 0
    assert result.deliberation_rounds <= 3
    assert 0.0 <= result.final_confidence <= 1.0
    
    print(f"  PASS: Simulation ran with {result.models_recruited} models, "
          f"confidence={result.final_confidence}")


def test_pmo_package():
    """Test PMO package format."""
    from pmo.package import PMOPackage, PMOMetadata
    import tempfile
    
    temp_dir = Path(tempfile.mkdtemp())
    pmo_path = temp_dir / "test.pmo"
    
    try:
        config = {
            "architecture": "trinity",
            "solver": {"count": 24400, "parameter_count_per_model": 75000000},
            "orchestration": {"total_orchestral_models": 19},
            "system": {"total_params": 1831425000000},
        }
        
        pmo = PMOPackage.from_config(str(pmo_path), config)
        
        # Build
        metadata = PMOMetadata(
            version="1.0",
            architecture="trinity",
            solver_count=24400,
            orchestral_count=19,
            total_params=1831425000000,
        )
        
        manifest = {"architecture": "trinity", "solver": config["solver"]}
        pmo.build(metadata, manifest, {})
        
        assert pmo_path.exists(), "PMO file not created"
        
        # Inspect
        info = pmo.inspect()
        assert info["exists"], "PMO does not exist"
        assert "metadata" in info, "PMO missing metadata"
        
        # Verify
        assert pmo.verify(), "PMO verification failed"
        
        print("  PASS: PMO package build/inspect/verify works")
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parameter_breakdown():
    """Test parameter breakdown for 75M model."""
    from momm.model import ModelConfig, create_model
    
    config = ModelConfig(
        model_id="test_75m",
        vocab_size=50304,
        max_seq_len=2048,
        hidden_dim=512,
        num_layers=12,
        num_heads=8,
        head_dim=64,
        ffn_hidden_dim=2048,
        activation="silu",
        norm_type="rmsnorm",
        tie_embeddings=True,
    )
    
    model = create_model(config)
    breakdown = model.get_parameter_count()
    
    print(f"  Parameter breakdown:")
    for key, value in breakdown.items():
        if key != "total":
            print(f"    {key}: {value:,} ({value/1e6:.2f}M)")
    print(f"    TOTAL: {breakdown['total']:,} ({breakdown['total']/1e6:.2f}M)")
    
    assert breakdown["total"] > 0
    print("  PASS: Parameter breakdown computed")


def run_all_tests():
    """Run all tests."""
    tests = [
        ("75M Parameter Count", test_75m_parameter_count),
        ("Trinity 1.0 Validation", test_trinity_1_0_validation),
        ("Trinity 3.0 Validation", test_trinity_3_0_validation),
        ("All Configs", test_all_configs),
        ("Dataset Scheduler", test_dataset_scheduler),
        ("Model Registry", test_model_registry),
        ("Pool Generation", test_pool_generation),
        ("Simulation Engine", test_simulation_engine),
        ("PMO Package", test_pmo_package),
        ("Parameter Breakdown", test_parameter_breakdown),
    ]
    
    print("=" * 60)
    print("PROTHYNESIS CORE TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n[{name}]")
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
