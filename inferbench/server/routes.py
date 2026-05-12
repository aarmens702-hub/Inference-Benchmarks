from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import RedirectResponse

from inferbench.engine.batcher import DynamicBatcher
from inferbench.engine.cache import PredictionCache
from inferbench.engine.controller import AdaptiveBatchController
from inferbench.server.inference import (
    InferConfig,
    InferOverloaded,
    InferTimedOut,
    run_inference,
)
from inferbench.server.registry import ModelEntry, ModelRegistry
from inferbench.server.schemas import (
    HealthResponse,
    InferRequest,
    InferResponse,
)


def _require_admin(authorization: str | None) -> None:
    token = os.environ.get("INFERBENCH_ADMIN_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="admin endpoints disabled")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="admin token required")


def _infer_config(request: Request) -> InferConfig:
    return InferConfig(
        cache=request.app.state.cache,
        request_timeout_ms=request.app.state.request_timeout_ms,
    )


async def _serve(entry: ModelEntry, payload: InferRequest, request: Request) -> InferResponse:
    try:
        return await run_inference(entry, payload.inputs, _infer_config(request), payload.request_id)
    except InferOverloaded as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except InferTimedOut as exc:
        raise HTTPException(status_code=504, detail=str(exc))


def register_routes(app: FastAPI) -> None:
    limiter = app.state.limiter
    rate = app.state.rate_limit_infer

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        # Send the bare root to the demo if it's mounted, otherwise /health.
        target = "/demo" if (app.state.config.get("demo", {}) or {}).get("enabled", True) else "/health"
        return RedirectResponse(url=target, status_code=307)

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        registry: ModelRegistry = request.app.state.registry
        default = registry.default()
        if default is None:
            raise HTTPException(status_code=503, detail="no models registered")
        runner = default.runner
        return HealthResponse(
            status="ok",
            model_id=runner.metadata.model_id,
            backend=runner.metadata.backend,
            precision=runner.metadata.precision,
        )

    @app.get("/version")
    def version(request: Request) -> dict:
        registry: ModelRegistry = request.app.state.registry
        return {
            "git_sha": request.app.state.git_sha,
            "models": [
                {
                    "name": e.name,
                    "precision": e.runner.metadata.precision,
                    "backend": e.runner.metadata.backend,
                }
                for e in registry.all()
            ],
        }

    async def infer_default(request: Request, payload: InferRequest) -> InferResponse:
        registry: ModelRegistry = request.app.state.registry
        entry = registry.default()
        if entry is None:
            raise HTTPException(status_code=503, detail="no models registered")
        return await _serve(entry, payload, request)

    async def infer_named(request: Request, model_name: str, payload: InferRequest) -> InferResponse:
        registry: ModelRegistry = request.app.state.registry
        entry = registry.get(model_name)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown model {model_name!r}; available: {registry.names()}",
            )
        return await _serve(entry, payload, request)

    if rate is not None:
        infer_default = limiter.limit(rate)(infer_default)
        infer_named = limiter.limit(rate)(infer_named)

    app.post("/infer", response_model=InferResponse)(infer_default)
    app.post("/infer/{model_name}", response_model=InferResponse)(infer_named)

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        cache: PredictionCache | None = request.app.state.cache
        registry: ModelRegistry = request.app.state.registry
        out: dict = {}
        if cache is not None:
            stats = cache.stats()
            out["cache"] = {
                "hits": stats.hits,
                "misses": stats.misses,
                "hit_ratio": stats.hit_ratio,
                "size": stats.size,
                "capacity": stats.capacity,
            }
        batchers_out: dict = {}
        for entry in registry.all():
            b: DynamicBatcher | None = entry.batcher
            if b is None:
                continue
            batchers_out[entry.name] = {
                "queue_size": b.qsize(),
                "avg_batch_size": b.avg_batch_size,
                "max_batch_size": b.config.max_batch_size,
                "max_wait_ms": b.config.max_wait_ms,
                "batch_size_histogram": {str(k): v for k, v in b.batch_size_histogram().items()},
            }
        if batchers_out:
            out["batchers"] = batchers_out
        controller: AdaptiveBatchController | None = request.app.state.controller
        if controller is not None:
            recent = controller.decisions[-10:]
            out["controller"] = {
                "tick": recent[-1].tick if recent else 0,
                "last_action": recent[-1].action if recent else "n/a",
                "last_p95_ms": recent[-1].p95_ms if recent else None,
                "recent_decisions": [
                    {
                        "tick": d.tick,
                        "action": d.action,
                        "p95_ms": d.p95_ms,
                        "qsize": d.qsize,
                        "max_batch_size": d.max_batch_size,
                        "max_wait_ms": d.max_wait_ms,
                    }
                    for d in recent
                ],
            }
        return out

    @app.post("/admin/cache/reset")
    def admin_cache_reset(request: Request, authorization: str | None = Header(None)) -> dict:
        _require_admin(authorization)
        cache: PredictionCache | None = request.app.state.cache
        if cache is None:
            raise HTTPException(status_code=404, detail="cache not enabled")
        cache.reset()
        return {"status": "ok", "size": 0}
