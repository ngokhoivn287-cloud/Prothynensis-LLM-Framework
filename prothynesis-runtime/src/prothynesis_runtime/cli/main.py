from __future__ import annotations

import argparse
import platform
import sys
from typing import Any

from prothynesis_runtime.runtime.orchestrator import RuntimeOrchestrator, InferenceEngine, ModelLoadError


def _hardware_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "cuda_available": False,
        "cuda_version": None,
        "gpu_name": None,
        "ram_gb": None,
        "torch_version": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass
    return info


def _print_dict(label: str, data: dict[str, Any]) -> None:
    print(f"{label}:")
    for k, v in data.items():
        print(f"  {k}: {v}")


def _get_orchestrator() -> RuntimeOrchestrator:
    return RuntimeOrchestrator()


def cmd_run(args: argparse.Namespace) -> int:
    """Run a single inference request."""
    try:
        engine = InferenceEngine(model_path=args.model, max_new_tokens=args.max_tokens, temperature=args.temperature)
        engine.load(args.model)
        response = engine.generate(args.prompt)
        print(response)
        return 0
    except ModelLoadError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


def cmd_chat(args: argparse.Namespace) -> int:
    orch = _get_orchestrator()
    if args.model:
        try:
            orch.inference_engine.load(args.model)
            orch._models = [{"id": args.model, "object": "model", "owned_by": "local", "capabilities": ["chat"]}]
        except ModelLoadError as exc:
            print(f"Error loading model: {exc}", file=sys.stderr)
            return 1
    print("Prothynesis chat. Type 'quit' to exit.")
    messages: list[dict[str, str]] = []
    model = args.model or "default"
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        try:
            result = orch.chat(messages, model_id=model)
            content = result.get("choices", [{}])[0].get("delta", {}).get("content", "")
        except ModelLoadError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            continue
        print(f"Assistant: {content}\n")
        messages.append({"role": "assistant", "content": content})
    return 0


def cmd_api(args: argparse.Namespace) -> int:
    import uvicorn

    from prothynesis_runtime.api.server import create_app

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    from prothynesis_runtime.web.app import create_app

    uvicorn.run(
        create_app(),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    orch = _get_orchestrator()
    _print_dict("Runtime", orch.get_status())
    return 0


def cmd_hardware(args: argparse.Namespace) -> int:
    _print_dict("Hardware", _hardware_info())
    return 0


def cmd_settings(args: argparse.Namespace) -> int:
    if args.action == "show":
        print("{'default_device': 'auto', 'max_new_tokens': 256, 'temperature': 0.7, 'top_p': 0.9}")
        return 0
    if args.action == "set":
        print(f"Set {args.key} (mock)")
        return 0
    if args.action == "recommend":
        print("Recommended concurrency: 1")
        return 0
    print("Unknown settings action")
    return 1


def cmd_model(args: argparse.Namespace) -> int:
    orch = _get_orchestrator()
    if args.action == "list":
        models = orch.get_models()
        if not models:
            print("No models loaded")
        else:
            for m in models:
                print(m["id"])
        return 0
    if args.action == "import":
        try:
            engine = InferenceEngine()
            engine.load(args.path)
            orch._models = [{"id": Path(args.path).stem, "object": "model", "owned_by": "local", "capabilities": ["chat"]}]
            print(f"Imported model: {Path(args.path).stem}")
            return 0
        except ModelLoadError as exc:
            print(f"Error importing model: {exc}", file=sys.stderr)
            return 1
    if args.action == "verify":
        from prothynesis_runtime.models.manager import RuntimeModelManager
        manager = RuntimeModelManager(registry_path="models/registry.json")
        result = manager.verify_model(args.path)
        print(result)
        return 0 if result.get("valid") else 1
    if args.action == "discover":
        from prothynesis_runtime.models.manager import RuntimeModelManager
        manager = RuntimeModelManager(registry_path="models/registry.json")
        models = manager.discover_models([args.path] if args.path else None)
        for m in models:
            print(f"{m.model_id} ({m.format}, {m.size_mb:.1f} MB)")
        return 0
    print(f"model {args.action} (not implemented)")
    return 1


def cmd_worker(args: argparse.Namespace) -> int:
    orch = _get_orchestrator()
    if args.action == "status":
        print(orch.get_worker_status())
        return 0
    if args.action == "start":
        print(orch.start_worker())
        return 0
    if args.action == "stop":
        print(orch.stop_worker())
        return 0
    print(f"worker {args.action} (not implemented)")
    return 1


def cmd_account(args: argparse.Namespace) -> int:
    orch = _get_orchestrator()
    if args.action == "status":
        print(orch.get_account_status())
        return 0
    if args.action == "connect":
        print(orch.connect_github())
        return 0
    if args.action == "disconnect":
        print(orch.disconnect_github())
        return 0
    print(f"account {args.action} (not implemented)")
    return 1


def cmd_doctor(args: argparse.Namespace) -> int:
    print("Prothynesis Runtime Doctor")
    print("-" * 40)
    hw = _hardware_info()
    _print_dict("Hardware", hw)
    orch = _get_orchestrator()
    doc = orch.run_doctor()
    _print_dict("Diagnostics", doc)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prothynesis", description="Prothynesis Runtime")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat")
    chat.add_argument("--model", default=None)

    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--temperature", type=float, default=0.7)

    api = sub.add_parser("api")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    web = sub.add_parser("web")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=3000)

    sub.add_parser("info")
    sub.add_parser("hardware")

    settings = sub.add_parser("settings")
    settings_sub = settings.add_subparsers(dest="action")
    settings_sub.add_parser("show")
    s_set = settings_sub.add_parser("set")
    s_set.add_argument("key")
    settings_sub.add_parser("recommend")

    model = sub.add_parser("model")
    model_sub = model.add_subparsers(dest="action")
    model_sub.add_parser("list")
    model_import = model_sub.add_parser("import")
    model_import.add_argument("path")
    model_verify = model_sub.add_parser("verify")
    model_verify.add_argument("path")
    model_discover = model_sub.add_parser("discover")
    model_discover.add_argument("path", nargs="?")

    worker = sub.add_parser("worker")
    worker_sub = worker.add_subparsers(dest="action")
    worker_sub.add_parser("status")
    worker_sub.add_parser("register")
    worker_sub.add_parser("login")
    worker_sub.add_parser("claim")
    worker_sub.add_parser("run")
    worker_sub.add_parser("validate")
    worker_sub.add_parser("submit")
    worker_sub.add_parser("stop")
    worker_sub.add_parser("history")
    worker_sub.add_parser("doctor")

    account = sub.add_parser("account")
    account_sub = account.add_subparsers(dest="action")
    account_sub.add_parser("status")
    account_sub.add_parser("connect")
    account_sub.add_parser("disconnect")

    sub.add_parser("doctor")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0

    commands = {
        "run": cmd_run,
        "chat": cmd_chat,
        "api": cmd_api,
        "web": cmd_web,
        "info": cmd_info,
        "hardware": cmd_hardware,
        "settings": cmd_settings,
        "model": cmd_model,
        "worker": cmd_worker,
        "account": cmd_account,
        "doctor": cmd_doctor,
    }

    handler = commands.get(args.command)
    if not handler:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
