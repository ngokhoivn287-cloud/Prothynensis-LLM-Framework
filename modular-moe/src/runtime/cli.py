"""CLI entry point for Prothynesis Runtime."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from src.runtime.controller import ProthynesisRuntime
from src.runtime.startup import FirstRunSetup, run_startup
from src.runtime.settings import SettingsManager
from src.utils.hardware import get_system_info


def cmd_runtime_info(args):
    """Show runtime info."""
    runtime = ProthynesisRuntime()
    info = runtime.get_runtime_info()
    import json
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_hardware(args):
    """Show hardware info."""
    from src.runtime.hardware import detect_hardware
    report = detect_hardware()
    import json
    print(json.dumps(report.to_dict(), indent=2, default=str))
    return 0


def cmd_first_run(args):
    """Run first-run setup."""
    settings = SettingsManager()
    setup = FirstRunSetup(settings)
    result = setup.run()
    print(setup.get_summary())
    return 0


def cmd_settings(args):
    """Show/modify settings."""
    settings = SettingsManager()
    
    if args.action == "show":
        import json
        print(json.dumps(settings.to_dict(), indent=2))
    elif args.action == "set" and args.key and args.value is not None:
        settings.set(args.key, args.value)
        print(f"Set {args.key} = {args.value}")
    elif args.action == "recommend":
        recs = settings.recommend_configuration()
        import json
        print(json.dumps(recs, indent=2))
    else:
        print("Usage: runtime settings show|set <key> <value>|recommend")
        return 1
    
    return 0


def cmd_model_list(args):
    """List models in registry."""
    runtime = ProthynesisRuntime()
    manager = runtime.model_manager
    models = manager.list_models()
    if not models:
        print("No models registered.")
        return 0
    print(f"{'Model ID':<20} {'Params':>10} {'Specialization':<20} {'Status'}")
    print("-" * 70)
    for m in models:
        print(f"{m['model_id']:<20} {m.get('parameter_count', 0):>10,} {m.get('specialization', 'general'):<20} {m.get('training_status', 'unknown')}")
    return 0


def cmd_model_import(args):
    """Import a model."""
    runtime = ProthynesisRuntime()
    result = runtime.model_manager.import_model(Path(args.path), args.model_id)
    if result["success"]:
        print(f"Imported model: {result['model_id']}")
        return 0
    else:
        print(f"Import failed: {result['errors']}")
        return 1


def cmd_model_verify(args):
    """Verify a model."""
    runtime = ProthynesisRuntime()
    result = runtime.model_manager.verify_model(Path(args.path))
    if result["valid"]:
        print(f"Valid: {args.path}")
        if result.get("metadata"):
            import json
            print(json.dumps(result["metadata"], indent=2))
        return 0
    else:
        print(f"Invalid: {result['errors']}")
        return 1


def cmd_model_discover(args):
    """Discover local models."""
    runtime = ProthynesisRuntime()
    models = runtime.model_manager.discover_models()
    if not models:
        print("No models found.")
        return 0
    for m in models:
        print(f"{m['path']} ({m['format']}, {m['size_mb']:.1f} MB)")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="prothynesis-runtime",
        description="Prothynesis Runtime CLI",
    )
    subparsers = parser.add_subparsers(dest="command")
    
    p_info = subparsers.add_parser("info", help="Show runtime info")
    p_info.set_defaults(func=cmd_runtime_info)
    
    p_hw = subparsers.add_parser("hardware", help="Show hardware info")
    p_hw.set_defaults(func=cmd_hardware)
    
    p_setup = subparsers.add_parser("first-run", help="Run first-run setup")
    p_setup.set_defaults(func=cmd_first_run)
    
    p_settings = subparsers.add_parser("settings", help="Manage settings")
    p_settings.add_argument("action", choices=["show", "set", "recommend"])
    p_settings.add_argument("key", nargs="?")
    p_settings.add_argument("value", nargs="?")
    p_settings.set_defaults(func=cmd_settings)
    
    p_model = subparsers.add_parser("model", help="Model management")
    model_sub = p_model.add_subparsers(dest="model_command")
    
    p_list = model_sub.add_parser("list", help="List models")
    p_list.set_defaults(func=cmd_model_list)
    
    p_import = model_sub.add_parser("import", help="Import model")
    p_import.add_argument("path", help="Path to model file")
    p_import.add_argument("--model-id", help="Model ID")
    p_import.set_defaults(func=cmd_model_import)
    
    p_verify = model_sub.add_parser("verify", help="Verify model")
    p_verify.add_argument("path", help="Path to model file")
    p_verify.set_defaults(func=cmd_model_verify)
    
    p_discover = model_sub.add_parser("discover", help="Discover local models")
    p_discover.set_defaults(func=cmd_model_discover)
    
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
