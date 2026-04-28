"""Route-layer tests: 429 overload, 504 timeout, cache hit/miss surfacing.

Spins up the FastAPI app with a fake ModelRunner so the test stays
fast and works without the exported ONNX model.
"""

from __future__ import annotations

import asyncio
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from inferbench.engine.batcher import BatcherConfig, DynamicBatcher
from inferbench.engine.cache import PredictionCache
from inferbench.engine.model_runner import ModelMetadata, Prediction, RunResult
from inferbench.server.routes import register_routes


class FakeRunner:
    def __init__(self, inference_ms: float = 1.0):
        self.inference_ms = inference_ms
        self.metadata = ModelMetadata(
            model_id="fake",
            precision="fp32",
            backend="onnxruntime-cpu",
            labels={0: "POSITIVE"},
            model_size_bytes=0,
        )

    def warmup(self, n: int = 0):
        return

    def run(self, texts):
        if self.inference_ms > 0:
            time.sleep(self.inference_ms / 1000.0)
        return RunResult(
            predictions=[Prediction(label="POSITIVE", score=0.9) for _ in texts],
            inference_ms=self.inference_ms,
            preprocess_ms=0.0,
            postprocess_ms=0.0,
            batch_size=len(texts),
        )


def _build_app(
    runner: FakeRunner,
    *,
    batcher_cfg: BatcherConfig | None = None,
    cache: PredictionCache | None = None,
    request_timeout_ms: float = 5000.0,
) -> tuple[FastAPI, DynamicBatcher | None]:
    """Construct an app without the YAML-driven lifespan, using a custom
    lifespan so tests inject precise components."""
    batcher: DynamicBatcher | None = None

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal batcher
        if batcher_cfg is not None:
            batcher = DynamicBatcher(runner, batcher_cfg)
            await batcher.start()
        app.state.runner = runner
        app.state.batcher = batcher
        app.state.cache = cache
        app.state.request_timeout_ms = request_timeout_ms
        try:
            yield
        finally:
            if batcher is not None:
                await batcher.stop()

    app = FastAPI(lifespan=lifespan)
    register_routes(app)
    return app, batcher


def test_health_returns_metadata():
    app, _ = _build_app(FakeRunner())
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_id"] == "fake"


def test_infer_no_batcher_no_cache():
    app, _ = _build_app(FakeRunner())
    with TestClient(app) as client:
        r = client.post("/infer", json={"inputs": ["hello"]})
        assert r.status_code == 200
        body = r.json()
        assert body["cache_hit"] is False
        assert body["batch_size"] == 1
        assert body["predictions"][0]["label"] == "POSITIVE"


def test_infer_cache_hit_skips_inference():
    runner = FakeRunner(inference_ms=10.0)
    cache = PredictionCache(capacity=64)
    app, _ = _build_app(runner, cache=cache)
    with TestClient(app) as client:
        # First call: miss
        r1 = client.post("/infer", json={"inputs": ["repeat"]})
        assert r1.status_code == 200
        assert r1.json()["cache_hit"] is False
        # Second call: hit
        r2 = client.post("/infer", json={"inputs": ["repeat"]})
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["cache_hit"] is True
        # Cache hit doesn't go through the runner -> batch_size==0
        assert body2["batch_size"] == 0


def test_infer_returns_429_when_queue_full():
    runner = FakeRunner(inference_ms=80.0)  # slow inference -> queue backs up
    # queue_max_size=2, max_wait_ms large so batches don't drain by timeout
    cfg = BatcherConfig(max_batch_size=1, max_wait_ms=1000.0, queue_max_size=2)
    app, _ = _build_app(runner, batcher_cfg=cfg, request_timeout_ms=10000.0)

    import httpx

    with TestClient(app) as client:
        # Fire many concurrent requests; once queue=2 fills, the 3rd+
        # rejects with 429 because put_nowait raises QueueOverflowError.
        async def fire(n: int):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as c:
                tasks = [c.post("/infer", json={"inputs": [f"q{i}"]}) for i in range(n)]
                return await asyncio.gather(*tasks, return_exceptions=True)

        responses = asyncio.run(fire(8))
        statuses = [r.status_code for r in responses if hasattr(r, "status_code")]
        assert 429 in statuses, f"expected at least one 429, got {statuses}"


def test_infer_returns_504_on_timeout():
    runner = FakeRunner(inference_ms=200.0)
    cfg = BatcherConfig(max_batch_size=8, max_wait_ms=10.0, queue_max_size=1024)
    # request_timeout_ms much shorter than the inference time
    app, _ = _build_app(runner, batcher_cfg=cfg, request_timeout_ms=20.0)
    with TestClient(app) as client:
        r = client.post("/infer", json={"inputs": ["slow"]})
        assert r.status_code == 504


def test_metrics_reports_cache_and_batcher_state():
    runner = FakeRunner()
    cache = PredictionCache(capacity=16)
    cfg = BatcherConfig(max_batch_size=4, max_wait_ms=10.0, queue_max_size=1024)
    app, _ = _build_app(runner, batcher_cfg=cfg, cache=cache)
    with TestClient(app) as client:
        client.post("/infer", json={"inputs": ["a"]})
        client.post("/infer", json={"inputs": ["a"]})  # cache hit
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["cache"]["hits"] == 1
        assert body["cache"]["misses"] == 1
        assert "batcher" in body
