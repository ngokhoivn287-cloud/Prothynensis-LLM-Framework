"""Configuration utilities for loading and merging YAML configs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def load_config_with_cli_overrides(
    config_path: str | Path,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config and apply CLI overrides."""
    config = load_config(config_path)
    if cli_overrides:
        config = merge_configs(config, cli_overrides)
    return config


def get_nested(config: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get a nested configuration value using dot notation."""
    keys = key.split(".")
    value = config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            return default
    return value


def set_nested(config: dict[str, Any], key: str, value: Any) -> None:
    """Set a nested configuration value using dot notation."""
    keys = key.split(".")
    current = config
    for k in keys[:-1]:
        if k not in current:
            current[k] = {}
        current = current[k]
    current[keys[-1]] = value


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save configuration to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def config_to_cli_args(config: dict[str, Any], prefix: str = "") -> list[str]:
    """Convert config dict to CLI argument list."""
    args = []
    for key, value in config.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            args.extend(config_to_cli_args(value, f"{full_key}."))
        elif isinstance(value, list):
            for v in value:
                args.append(f"--{full_key}")
                args.append(str(v))
        elif isinstance(value, bool):
            if value:
                args.append(f"--{full_key}")
        elif value is not None:
            args.append(f"--{full_key}")
            args.append(str(value))
    return args


def convert_config_types(config: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert config values to appropriate Python types.
    
    YAML loads all numbers as int/float, but some values may be strings
    that should be converted (e.g., '1e-5' -> 1e-5, 'true' -> True).
    """
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = convert_config_types(value)
        elif isinstance(value, list):
            result[key] = [convert_config_types(v) if isinstance(v, dict) else v for v in value]
        elif isinstance(value, str):
            # Try to convert string to appropriate type
            lower = value.lower()
            if lower in ('true', 'false'):
                result[key] = lower == 'true'
            else:
                # Try numeric conversion
                try:
                    if '.' in value or 'e' in lower:
                        result[key] = float(value)
                    else:
                        result[key] = int(value)
                except ValueError:
                    result[key] = value
        else:
            result[key] = value
    return result