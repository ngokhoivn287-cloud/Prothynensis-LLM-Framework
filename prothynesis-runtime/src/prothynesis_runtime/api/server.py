from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from prothynesis_runtime.runtime.orchestrator import RuntimeOrchestrator, ModelLoadError


def create_app(orchestrator: RuntimeOrchestrator | None = None) -> FastAPI:
    app = FastAPI(title="Prothynesis Runtime", version="0.1.0")
    orch = orchestrator or RuntimeOrchestrator()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        models = orch.get_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m["id"],
                    "object": m["object"],
                    "owned_by": m["owned_by"],
                    "capabilities": m.get("capabilities", []),
                }
                for m in models
            ],
        }

    @app.get("/v1/status")
    async def runtime_status() -> dict[str, Any]:
        return orch.get_status()

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: dict[str, Any]) -> dict[str, Any]:
        model = payload.get("model")
        if not model:
            raise HTTPException(status_code=400, detail="model is required")

        if not orch.inference_engine.loaded:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model}' is not loaded. Load a model first.",
            )

        messages = payload.get("messages", [])
        temperature = payload.get("temperature", 0.7)
        top_p = payload.get("top_p", 0.9)
        max_tokens = payload.get("max_tokens", 256)
        stream = payload.get("stream", False)

        if stream:
            return StreamingResponse(
                _stream_response(orch, model, messages, max_tokens),
                media_type="text/event-stream",
            )

        try:
            result = orch.chat(
                messages,
                model_id=model,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
            )
        except ModelLoadError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return result

    @app.get("/v1/worker/status")
    async def worker_status() -> dict[str, Any]:
        return orch.get_worker_status()

    @app.post("/v1/worker/start")
    async def worker_start() -> dict[str, str]:
        ok = orch.start_worker()
        return {"status": "started" if ok else "failed"}

    @app.post("/v1/worker/pause")
    async def worker_pause() -> dict[str, str]:
        ok = orch.pause_worker()
        return {"status": "paused" if ok else "failed"}

    @app.post("/v1/worker/stop")
    async def worker_stop() -> dict[str, str]:
        ok = orch.stop_worker()
        return {"status": "stopped" if ok else "failed"}

    @app.get("/v1/account")
    async def account_status() -> dict[str, Any]:
        return orch.get_account_status()

    @app.post("/v1/account/github/connect")
    async def github_connect() -> dict[str, str]:
        url = orch.connect_github()
        return {"auth_url": url}

    @app.post("/v1/account/github/disconnect")
    async def github_disconnect() -> dict[str, str]:
        ok = orch.disconnect_github()
        return {"status": "disconnected" if ok else "failed"}

    return app


async def _stream_response(
    orch: RuntimeOrchestrator,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> Any:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    try:
        result = orch.chat(messages, model_id=model, max_new_tokens=max_tokens)
        content = result.get("choices", [{}])[0].get("delta", {}).get("content", "")
    except ModelLoadError as exc:
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": f"Error: {exc}"}, "finish_reason": "error"}],
        }
        yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
        return

    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
    yield b"data: [DONE]\n\n"


app = create_app()
