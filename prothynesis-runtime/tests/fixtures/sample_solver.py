import hashlib
from pathlib import Path

SAMPLE_CHECKPOINT_DATA = b"fake checkpoint weights"

SAMPLE_VALIDATION_RESULT = {
    "passed": True,
    "score": 0.95,
    "loss": 0.05,
    "tokens_seen": 1_000_000,
}

SAMPLE_MANIFEST = {
    "model_id": "solver_001",
    "version": "0.1.0",
    "architecture": "cpu",
    "description": "Sample solver manifest",
    "solvers": [
        {
            "id": "solver_001",
            "path": "checkpoints/solver_001.bin",
            "specialization": "text-generation",
            "parameter_count": 7_000_000,
            "architecture": "cpu",
        }
    ],
    "orchestral_models": [],
    "system": {},
    "datasets": [],
    "training": {},
    "inference": {},
    "cache": {},
}


def write_sample_checkpoint(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SAMPLE_CHECKPOINT_DATA)
    return path
