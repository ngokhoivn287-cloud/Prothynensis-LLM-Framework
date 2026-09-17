import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SolverEntry:
    id: str
    path: str
    specialization: str
    parameter_count: int
    architecture: str


@dataclass
class PMOManifest:
    model_id: str = ""
    version: str = ""
    architecture: str = ""
    description: str = ""
    solvers: List[SolverEntry] = field(default_factory=list)
    orchestral_models: List[Dict[str, Any]] = field(default_factory=list)
    system: Dict[str, Any] = field(default_factory=dict)
    datasets: List[Dict[str, Any]] = field(default_factory=list)
    training: Dict[str, Any] = field(default_factory=dict)
    inference: Dict[str, Any] = field(default_factory=dict)
    cache: Dict[str, Any] = field(default_factory=dict)


def validate_manifest(data: Dict[str, Any]) -> tuple[bool, str]:
    required = {"model_id", "version", "architecture"}
    missing = required - data.keys()
    if missing:
        return False, f"Missing manifest keys: {', '.join(sorted(missing))}"
    if not isinstance(data.get("solvers"), list):
        return False, "'solvers' must be a list"
    for idx, solver in enumerate(data.get("solvers", [])):
        if not isinstance(solver, dict):
            return False, f"solver[{idx}] must be a dict"
        for key in ("id", "path", "specialization", "parameter_count", "architecture"):
            if key not in solver:
                return False, f"solver[{idx}] missing key '{key}'"
    if not isinstance(data.get("system"), dict):
        return False, "'system' must be a dict"
    return True, "valid"


def load_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
