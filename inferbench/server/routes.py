"""HTTP routes: /health and /infer (W1).

W3 will replace the inline runner.run() call with a queue submission, but
the public response schema stays stable.
"""

from __future__ import annotations

import time
import uuid

from fastapi import FastAPI, Request

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
    def infer(payload: InferRequest, request: Request) -> InferResponse:
        runner = request.app.state.runner
        request_id = payload.request_id or str(uuid.uuid4())

        t_start = time.perf_counter()
        # W1: no queue, no batching — direct call. queue_wait_ms == 0.
        result = runner.run(payload.inputs)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return InferResponse(
            request_id=request_id,
            predictions=[PredictionItem(label=p.label, score=p.score) for p in result.predictions],
            latency_ms=latency_ms,
            queue_wait_ms=0.0,
            inference_ms=result.inference_ms,
            batch_size=result.batch_size,
            cache_hit=False,
            model_backend=runner.metadata.backend,
            model_precision=runner.metadata.precision,
        )
