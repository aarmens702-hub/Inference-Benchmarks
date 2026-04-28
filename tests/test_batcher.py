"""DynamicBatcher correctness tests with a fake ModelRunner.

The fake records each call's batch size and an arrival timestamp so we
can assert flush-on-size and flush-on-timeout behavior without taking
on the cost of loading the real ONNX model.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest

from inferbench.engine.batcher import BatcherConfig, DynamicBatcher
from inferbench.engine.model_runner import ModelMetadata, Prediction, RunResult


@dataclass
class _RunCall:
    batch_size: int
    at: float


class FakeRunner:
    """Drop-in stand-in for ModelRunner with a configurable inference delay."""

    def __init__(self, inference_ms: float = 1.0) -> None:
        self.inference_ms = inference_ms
        self.calls: list[_RunCall] = []
        self.metadata = ModelMetadata(
            model_id="fake",
            precision="fp32",
            backend="onnxruntime-cpu",
            labels={0: "POSITIVE"},
            model_size_bytes=0,
        )

    def warmup(self, n: int = 0) -> None:
        return

    def run(self, texts):
        self.calls.append(_RunCall(batch_size=len(texts), at=time.perf_counter()))
        if self.inference_ms > 0:
            time.sleep(self.inference_ms / 1000.0)
        return RunResult(
            predictions=[Prediction(label="POSITIVE", score=0.99) for _ in texts],
            inference_ms=self.inference_ms,
            preprocess_ms=0.0,
            postprocess_ms=0.0,
            batch_size=len(texts),
        )


@pytest.mark.asyncio
async def test_flushes_on_max_batch_size():
    runner = FakeRunner(inference_ms=2.0)
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=4, max_wait_ms=10_000))
    await batcher.start()
    try:
        results = await asyncio.gather(*[batcher.submit(f"t{i}") for i in range(4)])
    finally:
        await batcher.stop()

    assert len(runner.calls) == 1
    assert runner.calls[0].batch_size == 4
    assert all(r.batch_size == 4 for r in results)


@pytest.mark.asyncio
async def test_flushes_on_max_wait_ms():
    runner = FakeRunner(inference_ms=1.0)
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=64, max_wait_ms=20))
    await batcher.start()
    try:
        # Submit 1 request — only flushes if max_wait_ms triggers.
        t0 = time.perf_counter()
        result = await batcher.submit("solo")
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        await batcher.stop()

    assert result.batch_size == 1
    assert len(runner.calls) == 1
    assert runner.calls[0].batch_size == 1
    # Should have waited ~max_wait_ms before the flush.
    assert 15 <= elapsed_ms < 200, f"expected ~20ms wait, got {elapsed_ms:.1f}ms"


@pytest.mark.asyncio
async def test_per_request_queue_wait_varies():
    """First member waits ~max_wait_ms; later arrivals wait less."""
    runner = FakeRunner(inference_ms=1.0)
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=8, max_wait_ms=50))
    await batcher.start()
    try:
        early = asyncio.create_task(batcher.submit("first"))
        await asyncio.sleep(0.030)  # 30 ms after first
        late = asyncio.create_task(batcher.submit("second"))
        results = await asyncio.gather(early, late)
    finally:
        await batcher.stop()

    early_r, late_r = results
    assert early_r.batch_size == 2
    assert late_r.batch_size == 2
    # First request waited longer than the late arrival.
    assert early_r.queue_wait_ms > late_r.queue_wait_ms
    # Both share the same inference_ms (one forward pass).
    assert early_r.inference_ms == late_r.inference_ms


@pytest.mark.asyncio
async def test_processes_multiple_consecutive_batches():
    runner = FakeRunner(inference_ms=1.0)
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=2, max_wait_ms=5))
    await batcher.start()
    try:
        # Submit 6 → expect 3 batches of 2.
        await asyncio.gather(*[batcher.submit(f"t{i}") for i in range(6)])
    finally:
        await batcher.stop()

    assert sum(c.batch_size for c in runner.calls) == 6
    assert all(c.batch_size <= 2 for c in runner.calls)


@pytest.mark.asyncio
async def test_runner_exception_propagates_to_all_batch_members():
    class BoomRunner(FakeRunner):
        def run(self, texts):
            raise RuntimeError("boom")

    runner = BoomRunner()
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=4, max_wait_ms=20))
    await batcher.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.gather(*[batcher.submit(f"t{i}") for i in range(4)])
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_avg_batch_size_tracks_workload():
    runner = FakeRunner(inference_ms=1.0)
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=8, max_wait_ms=10))
    await batcher.start()
    try:
        await asyncio.gather(*[batcher.submit(f"t{i}") for i in range(8)])
    finally:
        await batcher.stop()
    assert batcher.avg_batch_size > 0
