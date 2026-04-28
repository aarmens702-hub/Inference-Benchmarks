"""FastAPI application factory for InferBench.

Loads server configuration from configs/server.yaml (or $INFERBENCH_CONFIG),
constructs the ModelRunner at startup, and registers routes from routes.py.

W1 wires a synchronous /infer that calls ModelRunner.run directly. The
async batcher (W3) and queue/cache (W4) plug in here without changing the
public schema.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI

from inferbench.engine.model_runner import ModelRunner
from inferbench.server.routes import register_routes


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config_path = Path(os.environ.get("INFERBENCH_CONFIG", "configs/server.yaml"))
    config = _load_config(config_path)
    model_cfg = config["model"]

    runner = ModelRunner(
        model_dir=model_cfg["path"],
        backend=model_cfg["backend"],
    )
    runner.warmup(n=3)

    app.state.runner = runner
    app.state.config = config
    yield


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
