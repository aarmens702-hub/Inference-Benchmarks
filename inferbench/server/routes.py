"""HTTP routes: /health and /infer.

If a DynamicBatcher is registered on app.state, /infer submits each
input through the batcher (one queue item per text). Otherwise it
calls ModelRunner.run directly.

W4 adds three production concerns at the route layer:

- Cache lookup (cache_hit responses skip the queue entirely).
- 429 Too Many Requests when the bounded queue rejects an item.
- 504 Gateway Timeout when a request waits longer than
  request_timeout_ms.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import FastAPI, HTTPException, Request

from inferbench.engine.batcher import BatchResult, DynamicBatcher
from inferbench.engine.cache import PredictionCache, cache_key
from inferbench.engine.controller import AdaptiveBatchController
from inferbench.engine.request_queue import QueueOverflowError
from inferbench.server.schemas import (
    HealthResponse,
    InferRequest,
    InferResponse,
    PredictionItem,
)


def register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        runner = request.app.state.runner
        return HealthResponse(
            status="ok",
            model_id=runner.metadata.model_id,
            backend=runner.metadata.backend,
            precision=runner.metadata.precision,
        )

    @app.post("/infer", response_model=InferResponse)
    async def infer(payload: InferRequest, request: Request) -> InferResponse:
        runner = request.app.state.runner
        batcher: DynamicBatcher | None = request.app.state.batcher
        cache: PredictionCache | None = request.app.state.cache
        request_timeout_ms: float = request.app.state.request_timeout_ms

        request_id = payload.request_id or str(uuid.uuid4())
        t_start = time.perf_counter()

        # Reserve a slot per input (preserves request ordering).
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

    @app.get("/metrics")
    def metrics(request: Request) -> dict:
        cache: PredictionCache | None = request.app.state.cache
        batcher: DynamicBatcher | None = request.app.state.batcher
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
        if batcher is not None:
            out["batcher"] = {
                "queue_size": batcher.qsize(),
                "avg_batch_size": batcher.avg_batch_size,
                "max_batch_size": batcher.config.max_batch_size,
                "max_wait_ms": batcher.config.max_wait_ms,
                "batch_size_histogram": {str(k): v for k, v in batcher.batch_size_histogram().items()},
            }
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
    def admin_cache_reset(request: Request) -> dict:
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
    """Run the miss texts through the batcher (or directly through runner).

    Raises QueueOverflowError if the queue is full or asyncio.TimeoutError
    if any single submission exceeds timeout_sec.
    """
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
