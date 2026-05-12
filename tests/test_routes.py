"""Route-layer tests: registry routing, 429 overload, 504 timeout, cache, admin gate."""

from __future__ import annotations

import asyncio
import os
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from inferbench.engine.batcher import BatcherConfig, DynamicBatcher
from inferbench.engine.cache import PredictionCache
from inferbench.engine.model_runner import ModelMetadata, Prediction, RunResult
from inferbench.server.registry import ModelRegistry
from inferbench.server.routes import register_routes


class FakeRunner:
    def __init__(self, inference_ms: float = 1.0, model_id: str = "fake", precision: str = "fp32"):
        self.inference_ms = inference_ms
        self.metadata = ModelMetadata(
            model_id=model_id,
            precision=precision,
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
    runners: list[tuple[str, FakeRunner]] | None = None,
    *,
    runner: FakeRunner | None = None,
    batcher_cfg: BatcherConfig | None = None,
    cache: PredictionCache | None = None,
    request_timeout_ms: float = 5000.0,
    rate_limit: str | None = None,
) -> FastAPI:
    """Build an app with one or many models. Pass `runners=[(name, runner), ...]`
    for multi-model. Pass `runner=` for single-model (back-compat with old tests).
    """
    if runners is None:
        assert runner is not None, "pass either runners= or runner="
        runners = [("default", runner)]

    started: list[DynamicBatcher] = []

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = ModelRegistry()
        for i, (name, r) in enumerate(runners):
            batcher: DynamicBatcher | None = None
            if batcher_cfg is not None:
                batcher = DynamicBatcher(r, batcher_cfg)
                await batcher.start()
                started.append(batcher)
            registry.register(name, r, batcher, default=(i == 0))
        app.state.registry = registry
        app.state.cache = cache
        app.state.controller = None
        app.state.request_timeout_ms = request_timeout_ms
        app.state.git_sha = "testsha"
        app.state.rate_limit_infer = rate_limit
        default = registry.default()
        app.state.runner = default.runner if default else None
        app.state.batcher = default.batcher if default else None
        try:
            yield
        finally:
            for b in started:
                await b.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.limiter = Limiter(key_func=get_remote_address)
    app.state.rate_limit_infer = rate_limit
    app.state.git_sha = "testsha"
    app.add_exception_handler(
        RateLimitExceeded,
        lambda req, exc: JSONResponse(status_code=429, content={"detail": "rate limit exceeded"}),
    )
    register_routes(app)
    return app


def test_health_returns_metadata():
    app = _build_app(runner=FakeRunner())
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_id"] == "fake"


def test_infer_no_batcher_no_cache():
    app = _build_app(runner=FakeRunner())
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
    app = _build_app(runner=runner, cache=cache)
    with TestClient(app) as client:
        r1 = client.post("/infer", json={"inputs": ["repeat"]})
        assert r1.status_code == 200
        assert r1.json()["cache_hit"] is False
        r2 = client.post("/infer", json={"inputs": ["repeat"]})
        assert r2.status_code == 200
        body2 = r2.json()
        assert body2["cache_hit"] is True
        assert body2["batch_size"] == 0


def test_infer_returns_429_when_queue_full():
    runner = FakeRunner(inference_ms=80.0)
    cfg = BatcherConfig(max_batch_size=1, max_wait_ms=1000.0, queue_max_size=2)
    app = _build_app(runner=runner, batcher_cfg=cfg, request_timeout_ms=10000.0)

    import httpx

    with TestClient(app):
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
    app = _build_app(runner=runner, batcher_cfg=cfg, request_timeout_ms=20.0)
    with TestClient(app) as client:
        r = client.post("/infer", json={"inputs": ["slow"]})
        assert r.status_code == 504


def test_metrics_reports_cache_and_batcher_state():
    runner = FakeRunner()
    cache = PredictionCache(capacity=16)
    cfg = BatcherConfig(max_batch_size=4, max_wait_ms=10.0, queue_max_size=1024)
    app = _build_app(runner=runner, batcher_cfg=cfg, cache=cache)
    with TestClient(app) as client:
        client.post("/infer", json={"inputs": ["a"]})
        client.post("/infer", json={"inputs": ["a"]})
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.json()
        assert body["cache"]["hits"] == 1
        assert body["cache"]["misses"] == 1
        assert "batchers" in body and "default" in body["batchers"]


def test_infer_routes_to_named_model():
    fp32 = FakeRunner(model_id="m", precision="fp32")
    int8 = FakeRunner(model_id="m", precision="int8-dynamic")
    app = _build_app(runners=[("fp32", fp32), ("int8", int8)])
    with TestClient(app) as client:
        r1 = client.post("/infer/fp32", json={"inputs": ["x"]})
        r2 = client.post("/infer/int8", json={"inputs": ["x"]})
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["model_precision"] == "fp32"
        assert r2.json()["model_precision"] == "int8-dynamic"


def test_infer_unknown_model_returns_404():
    app = _build_app(runner=FakeRunner())
    with TestClient(app) as client:
        r = client.post("/infer/does-not-exist", json={"inputs": ["x"]})
        assert r.status_code == 404


def test_version_lists_models_and_git_sha():
    fp32 = FakeRunner(model_id="m", precision="fp32")
    int8 = FakeRunner(model_id="m", precision="int8-dynamic")
    app = _build_app(runners=[("fp32", fp32), ("int8", int8)])
    with TestClient(app) as client:
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert body["git_sha"] == "testsha"
        names = [m["name"] for m in body["models"]]
        assert names == ["fp32", "int8"]


def test_admin_cache_reset_503_without_token_env():
    cache = PredictionCache(capacity=8)
    app = _build_app(runner=FakeRunner(), cache=cache)
    old = os.environ.pop("INFERBENCH_ADMIN_TOKEN", None)
    try:
        with TestClient(app) as client:
            r = client.post("/admin/cache/reset")
            assert r.status_code == 503
    finally:
        if old is not None:
            os.environ["INFERBENCH_ADMIN_TOKEN"] = old


def test_admin_cache_reset_401_with_wrong_token():
    cache = PredictionCache(capacity=8)
    app = _build_app(runner=FakeRunner(), cache=cache)
    os.environ["INFERBENCH_ADMIN_TOKEN"] = "secret"
    try:
        with TestClient(app) as client:
            r = client.post("/admin/cache/reset", headers={"Authorization": "Bearer nope"})
            assert r.status_code == 401
    finally:
        os.environ.pop("INFERBENCH_ADMIN_TOKEN", None)


def test_admin_cache_reset_200_with_correct_token():
    cache = PredictionCache(capacity=8)
    app = _build_app(runner=FakeRunner(), cache=cache)
    os.environ["INFERBENCH_ADMIN_TOKEN"] = "secret"
    try:
        with TestClient(app) as client:
            r = client.post("/admin/cache/reset", headers={"Authorization": "Bearer secret"})
            assert r.status_code == 200
    finally:
        os.environ.pop("INFERBENCH_ADMIN_TOKEN", None)


def test_rate_limit_429_after_threshold():
    app = _build_app(runner=FakeRunner(), rate_limit="3/minute")
    with TestClient(app) as client:
        statuses = [client.post("/infer", json={"inputs": ["a"]}).status_code for _ in range(6)]
        assert 200 in statuses
        assert 429 in statuses
