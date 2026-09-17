from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from prothynesis_runtime.runtime.orchestrator import RuntimeOrchestrator


def create_app(orchestrator: RuntimeOrchestrator | None = None) -> FastAPI:
    app = FastAPI(title="Prothynesis Web UI")
    orch = orchestrator or RuntimeOrchestrator()

    base_dir = Path(__file__).resolve().parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    if templates_dir.exists():
        templates = Jinja2Templates(directory=str(templates_dir))
    else:
        templates = None

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        context = _common_context(request)
        return templates.TemplateResponse("index.html", context)

    @app.get("/models", response_class=HTMLResponse)
    async def models_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        context["models"] = orch.get_models()
        return templates.TemplateResponse("models.html", context)

    @app.get("/worker", response_class=HTMLResponse)
    async def worker_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        context["worker_status"] = orch.get_worker_status()
        return templates.TemplateResponse("worker.html", context)

    @app.get("/account", response_class=HTMLResponse)
    async def account_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        context["account_status"] = orch.get_account_status()
        return templates.TemplateResponse("account.html", context)

    @app.get("/hardware", response_class=HTMLResponse)
    async def hardware_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        return templates.TemplateResponse("hardware.html", context)

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        return templates.TemplateResponse("settings.html", context)

    @app.get("/logs", response_class=HTMLResponse)
    async def logs_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        return templates.TemplateResponse("logs.html", context)

    @app.get("/runtime", response_class=HTMLResponse)
    async def runtime_page(request: Request) -> HTMLResponse:
        context = _common_context(request)
        return templates.TemplateResponse("runtime.html", context)

    return app


def _common_context(request: Request) -> dict[str, Any]:
    return {
        "request": request,
        "runtime_info": {"version": "0.1.0", "running": True},
        "hardware_info": {},
        "models": [],
        "worker_status": {"status": "IDLE", "current_solver": None, "progress": 0.0},
        "account_status": {"connected": False, "username": None},
    }


app = create_app()
