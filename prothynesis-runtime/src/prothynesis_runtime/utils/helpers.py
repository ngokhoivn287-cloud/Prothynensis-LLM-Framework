from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def generate_solver_id(seed: int) -> str:
    return f"{seed:07d}"


def compute_job_hash(job_dict: dict) -> str:
    canonical = canonical_json(job_dict)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_training_config_hash(config: dict) -> str:
    canonical = canonical_json(config)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directory(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_json_load(path: str | Path) -> tuple[Optional[dict], Optional[str]]:
    p = Path(path)
    if not p.exists():
        return None, f"File not found: {path}"
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode error: {e}"
    except Exception as e:
        return None, str(e)


def safe_json_dump(data: dict, path: str | Path) -> bool:
    p = Path(path)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
