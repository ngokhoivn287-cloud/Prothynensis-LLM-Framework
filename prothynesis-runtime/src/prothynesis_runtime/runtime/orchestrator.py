from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

import torch

from prothynesis_runtime.runtime.engine import (
    ModelLoadError,
    UnsupportedFormatError,
    load_model_from_pmo,
    load_checkpoint,
    _build_model_from_checkpoint,
    _resolve_device,
)


class InferenceEngine:
    def __init__(
        self,
        model_path: str | None = None,
        device: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> None:
        self.model_path = model_path
        self.device = _resolve_device(device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False
        self._last_error: str | None = None
        self._vocab_size: int = 1024

    @property
    def loaded(self) -> bool:
        return self._loaded and self._model is not None

    def _ensure_loaded(self) -> None:
        if self._loaded and self._model is not None:
            return
        if self.model_path is None:
            raise ModelLoadError("No model path configured")
        self.load(self.model_path)

    def load(self, path: str) -> None:
        self.unload()
        path_obj = Path(path)
        if not path_obj.exists():
            raise ModelLoadError(f"Model path not found: {path}")

        try:
            is_pmo = path_obj.suffix.lower() == ".pmo"
            is_zip = zipfile.is_zipfile(str(path_obj)) and not is_pmo and path_obj.suffix.lower() not in (".pt", ".bin", ".pth")
            if is_pmo or is_zip:
                result = load_model_from_pmo(path_obj, device=self.device)
                if len(result) == 3:
                    self._model, vocab_size, _ = result
                else:
                    self._model = result[0]
                    vocab_size = 1024
            else:
                self._model, vocab_size = _build_model_from_checkpoint(path_obj, device=self.device)
            self._vocab_size = vocab_size
            self._loaded = True
            self._last_error = None
        except Exception as exc:
            self.unload()
            self._last_error = str(exc)
            raise ModelLoadError(f"Failed to load model: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        if self._tokenizer is None:
            from prothynesis_runtime.runtime.engine import SimpleCharacterTokenizer
            self._tokenizer = SimpleCharacterTokenizer(vocab_size=self._vocab_size)
        tokenizer = self._tokenizer
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        model_device = next(self._model.parameters()).device if self._model is not None else torch.device(self.device)
        input_ids = input_ids.to(model_device)
        with torch.no_grad():
            outputs = self._model(input_ids)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        return tokenizer.decode(next_token[0].tolist())

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not messages:
            return ""
        last = messages[-1].get("content", "")
        return self.generate(last)

    def get_last_error(self) -> str | None:
        return self._last_error


class RuntimeOrchestrator:
    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        model_manager: Any = None,
        inference_engine: InferenceEngine | None = None,
        worker_client: Any = None,
        api_server: Any = None,
        web_app: Any = None,
    ) -> None:
        self.settings = settings or {}
        self.model_manager = model_manager
        self.inference_engine = inference_engine or InferenceEngine()
        self.worker_client = worker_client
        self.api_server = api_server
        self.web_app = web_app
        self._models: list[dict[str, Any]] = []
        self._worker_status: dict[str, Any] = {
            "status": "IDLE",
            "current_solver": None,
            "progress": 0.0,
            "gpu": {},
            "vram": None,
            "tokens": 0,
        }
        self._github: dict[str, Any] = {"connected": False, "username": None}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.inference_engine.unload()

    def get_status(self) -> dict[str, Any]:
        return {
            "running": True,
            "models_loaded": len(self._models),
            "inference_loaded": self.inference_engine.loaded,
            "device": self.inference_engine.device,
            "hardware": self._get_hardware_info(),
        }

    def get_models(self) -> list[dict[str, Any]]:
        return list(self._models)

    def chat(
        self, messages: list[dict[str, str]], model_id: str, **generation_kwargs: Any
    ) -> dict[str, Any]:
        if not self.inference_engine.loaded:
            raise ModelLoadError(
                f"Model not loaded. Load a model first. Last error: {self.inference_engine.get_last_error()}"
            )
        response = self.inference_engine.chat(messages)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": response},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(m.get("content", "").split()) for m in messages),
                "completion_tokens": len(response.split()),
                "total_tokens": sum(len(m.get("content", "").split()) for m in messages) + len(response.split()),
            },
        }

    def start_worker(self) -> bool:
        self._worker_status["status"] = "RUNNING"
        return True

    def pause_worker(self) -> bool:
        if self._worker_status["status"] == "RUNNING":
            self._worker_status["status"] = "PAUSED"
            return True
        return False

    def stop_worker(self) -> bool:
        self._worker_status["status"] = "IDLE"
        self._worker_status["current_solver"] = None
        self._worker_status["progress"] = 0.0
        return True

    def get_worker_status(self) -> dict[str, Any]:
        return dict(self._worker_status)

    def connect_github(self) -> str:
        auth_url = f"https://github.com/login/oauth/authorize?client_id=mock&state={uuid.uuid4().hex}"
        self._github["connected"] = True
        self._github["username"] = "mockuser"
        return auth_url

    def disconnect_github(self) -> bool:
        self._github = {"connected": False, "username": None}
        return True

    def get_account_status(self) -> dict[str, Any]:
        return dict(self._github)

    def run_doctor(self) -> dict[str, Any]:
        hw = self._get_hardware_info()
        return {
            "python_version": "3.11+",
            "torch_version": "2.x",
            "cuda_available": hw.get("cuda_available", False),
            "cuda_version": hw.get("cuda_version"),
            "gpu_info": hw.get("gpu_name"),
            "ram_gb": hw.get("ram_gb"),
            "runtime_version": "0.1.0",
            "estimated_solver_capability": "real" if self.inference_engine.loaded else "none",
            "recommended_concurrency": 1,
            "inference_loaded": self.inference_engine.loaded,
            "last_error": self.inference_engine.get_last_error(),
        }

    def _get_hardware_info(self) -> dict[str, Any]:
        try:
            import torch

            cuda_available = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
            cuda_version = torch.version.cuda if cuda_available else None
        except Exception:
            cuda_available = False
            gpu_name = None
            cuda_version = None

        try:
            import psutil

            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception:
            ram_gb = None

        return {
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "gpu_name": gpu_name,
            "ram_gb": ram_gb,
        }
