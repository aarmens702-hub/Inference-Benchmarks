from __future__ import annotations

import asyncio
import os
import time
import uuid

from fastapi import FastAPI, Header, HTTPException, Request

from inferbench.engine.batcher import BatchResult, DynamicBatcher
from inferbench.engine.cache import PredictionCache, cache_key
from inferbench.engine.controller import AdaptiveBatchController
from inferbench.engine.request_queue import QueueOverflowError
from inferbench.server.registry import ModelEntry, ModelRegistry
from inferbench.server.schemas import (
    HealthResponse,
    InferRequest,
    InferResponse,
    PredictionItem,
)


def _require_admin(authorization: str | None) -> None:
    token = os.environ.get("INFERBENCH_ADMIN_TOKEN")
    if not token:
        raise HTTPException(status_code=503, detail="admin endpoints disabled")
    if authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="admin token required")


def register_routes(app: FastAPI) -> None:
    limiter = app.state.limiter

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

    rate = app.state.rate_limit_infer

    async def _infer_impl(entry: ModelEntry, payload: InferRequest, request: Request) -> InferResponse:
        cache: PredictionCache | None = request.app.state.cache
        request_timeout_ms: float = request.app.state.request_timeout_ms

        runner = entry.runner
        batcher = entry.batcher
        request_id = payload.request_id or str(uuid.uuid4())
        t_start = time.perf_counter()

        n = len(payload.inputs)
        predictions: list[PredictionItem | None] = [None] * n
        miss_indices: list[int] = []
        miss_texts: list[str] = []
        cache_hits_count = 0

        if cache is not None:
            for i, text in enumerate(payload.inputs):
                key = cache_key(runner.metadata.model_id, runner.metadata.precision, text)
                cached = cache.get(key)
                if cached is not None:
                    predictions[i] = PredictionItem(label=cached.label, score=cached.score)
                    cache_hits_count += 1
                else:
                    miss_indices.append(i)
                    miss_texts.append(text)
        else:
            miss_indices = list(range(n))
            miss_texts = list(payload.inputs)

        batch_size = 0
        queue_wait_ms = 0.0
        inference_ms = 0.0

        if miss_texts:
            try:
                batch_results = await _submit_misses(
                    batcher=batcher,
                    runner=runner,
                    miss_texts=miss_texts,
                    timeout_sec=request_timeout_ms / 1000.0,
                )
            except QueueOverflowError:
                raise HTTPException(status_code=429, detail="queue full")
            except asyncio.TimeoutError:
                raise HTTPException(status_code=504, detail="request timed out")

            batch_size = batch_results[0].batch_size if batch_results else 0
            queue_wait_ms = max((r.queue_wait_ms for r in batch_results), default=0.0)
            inference_ms = batch_results[0].inference_ms if batch_results else 0.0

            for j, idx in enumerate(miss_indices):
                br = batch_results[j]
                predictions[idx] = PredictionItem(label=br.prediction.label, score=br.prediction.score)
                if cache is not None:
                    key = cache_key(runner.metadata.model_id, runner.metadata.precision, miss_texts[j])
                    cache.put(key, br.prediction)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return InferResponse(
            request_id=request_id,
            predictions=predictions,  # type: ignore[arg-type]
            latency_ms=latency_ms,
            queue_wait_ms=queue_wait_ms,
            inference_ms=inference_ms,
            batch_size=batch_size if miss_texts else 0,
            cache_hit=(cache_hits_count == n and n > 0),
            model_backend=runner.metadata.backend,
            model_precision=runner.metadata.precision,
        )

    async def infer_default(request: Request, payload: InferRequest) -> InferResponse:
        registry: ModelRegistry = request.app.state.registry
        entry = registry.default()
        if entry is None:
            raise HTTPException(status_code=503, detail="no models registered")
        return await _infer_impl(entry, payload, request)

    async def infer_named(request: Request, model_name: str, payload: InferRequest) -> InferResponse:
        registry: ModelRegistry = request.app.state.registry
        entry = registry.get(model_name)
        if entry is None:
            raise HTTPException(
                status_code=404,
                detail=f"unknown model {model_name!r}; available: {registry.names()}",
            )
        return await _infer_impl(entry, payload, request)

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
        # One batcher per model — surface them as a map.
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


async def _submit_misses(
    batcher: DynamicBatcher | None,
    runner,
    miss_texts: list[str],
    timeout_sec: float,
) -> list[BatchResult]:
    if batcher is not None:
        async def _one(t: str) -> BatchResult:
            return await asyncio.wait_for(batcher.submit(t), timeout=timeout_sec)

        return await asyncio.gather(*[_one(t) for t in miss_texts])

    run_result = runner.run(miss_texts)
    return [
        BatchResult(
            prediction=p,
            batch_size=run_result.batch_size,
            queue_wait_ms=0.0,
            inference_ms=run_result.inference_ms,
        )
        for p in run_result.predictions
    ]
