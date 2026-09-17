SAMPLE_JOB_VALID = {
    "job_id": "job_2024_0000001",
    "solver_id": "1234567",
    "runtime_version": "0.1.0",
    "model_architecture": "cpu",
    "seed": 42,
    "dataset_id": "ds_001",
    "token_budget": 10_000_000,
    "sequence_length": 1024,
    "training_config_hash": "abc123",
    "validation_config_hash": "def456",
    "job_hash": "ghi789",
    "signature": "sig123",
    "created_at": "2024-01-01T00:00:00",
    "expires_at": "2025-01-01T00:00:00",
}

SAMPLE_JOB_INVALID_MISSING_FIELDS = {
    "job_id": "job_2024_0000001",
}

SAMPLE_JOB_INVALID_SIGNATURE = {
    "job_id": "job_2024_0000002",
    "solver_id": "7654321",
    "runtime_version": "0.1.0",
    "model_architecture": "cpu",
    "seed": 42,
    "dataset_id": "ds_002",
    "token_budget": 10_000_000,
    "sequence_length": 1024,
    "training_config_hash": "abc123",
    "validation_config_hash": "def456",
    "job_hash": "ghi789",
    "signature": "",
    "created_at": "2024-01-01T00:00:00",
}

SAMPLE_TRAINING_CONFIG = {
    "model_architecture": "cpu",
    "seed": 42,
    "dataset_id": "ds_001",
    "token_budget": 10_000_000,
    "sequence_length": 1024,
    "learning_rate": 0.001,
    "batch_size": 32,
    "gradient_accumulation_steps": 1,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "optimizer": "adamw",
    "scheduler": "cosine",
}

SAMPLE_VALIDATION_CONFIG = {
    "validation_steps": 6,
    "validation_prompts": ["Hello world"],
    "max_new_tokens": 256,
    "temperature": 0.7,
    "top_p": 0.9,
}


def make_sample_job(overrides=None):
    job = dict(SAMPLE_JOB_VALID)
    if overrides:
        job.update(overrides)
    return job
