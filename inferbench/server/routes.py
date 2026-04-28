"""HTTP routes: /health and /infer.

If a DynamicBatcher is registered on app.state, /infer submits each
input through the batcher (one queue item per text). Otherwise it
calls ModelRunner.run directly (W1 fast path, also useful for the
no-batching baseline).
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import FastAPI, Request

from inferbench.engine.batcher import DynamicBatcher
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
        request_id = payload.request_id or str(uuid.uuid4())

        t_start = time.perf_counter()

        if batcher is not None:
            results = await asyncio.gather(*[batcher.submit(t) for t in payload.inputs])
            predictions = [PredictionItem(label=r.prediction.label, score=r.prediction.score) for r in results]
            # When a single client request fans out to multiple queue items,
            # report the worst per-item queue wait and the (shared) inference
            # time of the batch the first item landed in.
            batch_size = results[0].batch_size if results else 0
            queue_wait_ms = max(r.queue_wait_ms for r in results) if results else 0.0
            inference_ms = results[0].inference_ms if results else 0.0
        else:
            run_result = runner.run(payload.inputs)
            predictions = [PredictionItem(label=p.label, score=p.score) for p in run_result.predictions]
            batch_size = run_result.batch_size
            queue_wait_ms = 0.0
            inference_ms = run_result.inference_ms

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return InferResponse(
            request_id=request_id,
            predictions=predictions,
            latency_ms=latency_ms,
            queue_wait_ms=queue_wait_ms,
            inference_ms=inference_ms,
            batch_size=batch_size,
            cache_hit=False,
            model_backend=runner.metadata.backend,
            model_precision=runner.metadata.precision,
        )
