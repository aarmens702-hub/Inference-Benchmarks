from __future__ import annotations

import os
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from inferbench.engine.batcher import BatcherConfig, DynamicBatcher
from inferbench.engine.cache import PredictionCache
from inferbench.engine.controller import AdaptiveBatchController, ControllerConfig
from inferbench.engine.model_runner import ModelRunner
from inferbench.server.registry import ModelRegistry
from inferbench.server.routes import register_routes


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _model_specs(config: dict) -> list[dict]:
    if "models" in config:
        return list(config["models"])
    if "model" in config:
        m = dict(config["model"])
        m.setdefault("name", "default")
        return [m]
    raise ValueError("config has neither 'models' nor 'model'")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


_GIT_SHA = _git_sha()


def _build_lifespan(config: dict):
    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        batching_cfg = config.get("batching", {}) or {}
        queue_cfg = config.get("queue", {}) or {}
        cache_cfg = config.get("cache", {}) or {}
        controller_cfg = config.get("controller", {}) or {}

        skip_warmup = os.environ.get("INFERBENCH_SKIP_WARMUP") == "1"

        cache: PredictionCache | None = None
        if cache_cfg.get("enabled", False):
            cache = PredictionCache(capacity=int(cache_cfg.get("max_entries", 4096)))

        registry = ModelRegistry()
        started_batchers: list[DynamicBatcher] = []

        for i, spec in enumerate(_model_specs(config)):
            runner = ModelRunner(model_dir=spec["path"], backend=spec["backend"])
            if not skip_warmup:
                runner.warmup(n=3)

            batcher: DynamicBatcher | None = None
            if batching_cfg.get("enabled", False):
                batcher = DynamicBatcher(
                    runner,
                    BatcherConfig(
                        max_batch_size=int(batching_cfg.get("max_batch_size", 16)),
                        max_wait_ms=float(batching_cfg.get("max_wait_ms", 10.0)),
                        queue_max_size=int(queue_cfg.get("max_size", 0)),
                    ),
                )
                await batcher.start()
                started_batchers.append(batcher)

            registry.register(spec["name"], runner, batcher, default=(i == 0))

        controller: AdaptiveBatchController | None = None
        if controller_cfg.get("enabled", False):
            default = registry.default()
            if default is not None and default.batcher is not None:
                controller = AdaptiveBatchController(
                    default.batcher,
                    ControllerConfig(
                        latency_slo_ms=float(controller_cfg.get("latency_slo_ms", 100.0)),
                        adjust_interval_sec=float(controller_cfg.get("adjust_interval_sec", 5.0)),
                        min_batch_size=int(controller_cfg.get("min_batch_size", 1)),
                        max_batch_size=int(controller_cfg.get("max_batch_size", 32)),
                    ),
                )
                await controller.start()

        app.state.registry = registry
        app.state.cache = cache
        app.state.controller = controller
        app.state.request_timeout_ms = float(queue_cfg.get("request_timeout_ms", 5000))
        app.state.config = config
        default = registry.default()
        # Back-compat aliases for tests/benchmarks that read these directly.
        app.state.runner = default.runner if default else None
        app.state.batcher = default.batcher if default else None

        try:
            yield
        finally:
            if controller is not None:
                await controller.stop()
            for b in started_batchers:
                await b.stop()

    return _lifespan


def _rate_limit_handler(request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})


def create_app() -> FastAPI:
    config_path = Path(os.environ.get("INFERBENCH_CONFIG", "configs/server.yaml"))
    config = _load_config(config_path)
    rate_limit_cfg = config.get("rate_limit", {}) or {}
    rate_limit_infer = (
        rate_limit_cfg.get("infer", "30/minute")
        if rate_limit_cfg.get("enabled", True)
        else None
    )

    app = FastAPI(
        title="InferBench",
        version="0.1.0",
        description="Local inference-serving benchmark framework.",
        lifespan=_build_lifespan(config),
    )
    app.state.limiter = Limiter(key_func=get_remote_address)
    app.state.rate_limit_infer = rate_limit_infer
    app.state.git_sha = _GIT_SHA
    app.add_exception_handler(RateLimitExceeded, _rate_limit_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routes(app)
    return app


app = create_app()
