"""FastAPI application factory for InferBench.

Loads server configuration from configs/server.yaml (or $INFERBENCH_CONFIG),
constructs the ModelRunner at startup, and (when batching.enabled is true)
starts a DynamicBatcher background task. Routes auto-detect whether a
batcher is present.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI

from inferbench.engine.batcher import BatcherConfig, DynamicBatcher
from inferbench.engine.cache import PredictionCache
from inferbench.engine.controller import AdaptiveBatchController, ControllerConfig
from inferbench.engine.model_runner import ModelRunner
from inferbench.server.routes import register_routes


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config_path = Path(os.environ.get("INFERBENCH_CONFIG", "configs/server.yaml"))
    config = _load_config(config_path)
    model_cfg = config["model"]
    batching_cfg = config.get("batching", {}) or {}
    queue_cfg = config.get("queue", {}) or {}
    cache_cfg = config.get("cache", {}) or {}

    runner = ModelRunner(
        model_dir=model_cfg["path"],
        backend=model_cfg["backend"],
    )
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

    cache: PredictionCache | None = None
    if cache_cfg.get("enabled", False):
        cache = PredictionCache(capacity=int(cache_cfg.get("max_entries", 4096)))

    controller: AdaptiveBatchController | None = None
    controller_cfg = config.get("controller", {}) or {}
    if controller_cfg.get("enabled", False) and batcher is not None:
        controller = AdaptiveBatchController(
            batcher,
            ControllerConfig(
                latency_slo_ms=float(controller_cfg.get("latency_slo_ms", 100.0)),
                adjust_interval_sec=float(controller_cfg.get("adjust_interval_sec", 5.0)),
                min_batch_size=int(controller_cfg.get("min_batch_size", 1)),
                max_batch_size=int(controller_cfg.get("max_batch_size", 32)),
            ),
        )
        await controller.start()

    app.state.runner = runner
    app.state.batcher = batcher
    app.state.cache = cache
    app.state.controller = controller
    app.state.request_timeout_ms = float(queue_cfg.get("request_timeout_ms", 5000))
    app.state.config = config

    try:
        yield
    finally:
        if controller is not None:
            await controller.stop()
        if batcher is not None:
            await batcher.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="InferBench",
        version="0.1.0",
        description="Local inference-serving benchmark framework.",
        lifespan=_lifespan,
    )
    register_routes(app)
    return app


app = create_app()
