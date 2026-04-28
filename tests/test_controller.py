"""AdaptiveBatchController decision logic tests.

Uses step_once() to exercise one tick at a time, deterministically,
without waiting on the asyncio loop.
"""

from __future__ import annotations

from collections import deque

import pytest

from inferbench.engine.batcher import BatcherConfig, DynamicBatcher
from inferbench.engine.controller import (
    AdaptiveBatchController,
    ControllerConfig,
    _percentile,
)
from inferbench.engine.model_runner import ModelMetadata, Prediction, RunResult


class FakeRunner:
    def __init__(self):
        self.metadata = ModelMetadata(
            model_id="fake",
            precision="fp32",
            backend="onnxruntime-cpu",
            labels={0: "POSITIVE"},
            model_size_bytes=0,
        )

    def warmup(self, n=0):
        return

    def run(self, texts):
        return RunResult(
            predictions=[Prediction("POSITIVE", 0.99) for _ in texts],
            inference_ms=1.0,
            preprocess_ms=0.0,
            postprocess_ms=0.0,
            batch_size=len(texts),
        )


def _seed_batcher(latencies: list[float], qsize_target: int = 0) -> DynamicBatcher:
    """Build a batcher with synthetic recent_latencies and qsize.

    We don't actually start the batcher task — controller.step_once()
    only reads recent_latencies() and qsize(), both of which are public.
    """
    runner = FakeRunner()
    batcher = DynamicBatcher(runner, BatcherConfig(max_batch_size=8, max_wait_ms=10.0))
    batcher._recent_latencies = deque(latencies, maxlen=2048)

    class _FakeQ:
        def __init__(self, n: int) -> None:
            self.n = n

        def qsize(self) -> int:
            return self.n

    batcher._queue = _FakeQ(qsize_target)  # type: ignore[assignment]
    return batcher


def test_no_data_tick_when_under_min_samples():
    batcher = _seed_batcher(latencies=[5.0] * 10)
    ctrl = AdaptiveBatchController(batcher, ControllerConfig(min_samples=32))
    decision = ctrl.step_once()
    assert decision.action == "no-data"
    assert decision.max_batch_size == 8


def test_shrinks_when_p95_exceeds_slo():
    # latencies cluster at 200ms — well over the 100ms slo.
    batcher = _seed_batcher(latencies=[200.0] * 100)
    ctrl = AdaptiveBatchController(
        batcher,
        ControllerConfig(latency_slo_ms=100.0, batch_step=2, wait_step_ms=2.0),
    )
    decision = ctrl.step_once()
    assert decision.action == "shrink"
    assert decision.max_batch_size == 6  # was 8, stepped down 2
    assert decision.max_wait_ms == 8.0


def test_grows_when_under_slo_and_queue_full():
    # latencies well under SLO, queue depth high.
    batcher = _seed_batcher(latencies=[10.0] * 100, qsize_target=8)
    ctrl = AdaptiveBatchController(batcher, ControllerConfig(latency_slo_ms=100.0))
    decision = ctrl.step_once()
    assert decision.action == "grow"
    assert decision.max_batch_size == 10  # was 8, stepped up 2
    assert decision.max_wait_ms == 12.0


def test_holds_when_under_slo_but_queue_empty():
    # No queue pressure -> don't grow even if latency is low.
    batcher = _seed_batcher(latencies=[10.0] * 100, qsize_target=0)
    ctrl = AdaptiveBatchController(batcher, ControllerConfig(latency_slo_ms=100.0))
    decision = ctrl.step_once()
    assert decision.action == "hold"
    assert decision.max_batch_size == 8
    assert decision.max_wait_ms == 10.0


def test_clamps_at_min_batch_size():
    batcher = _seed_batcher(latencies=[500.0] * 100)
    batcher.config.max_batch_size = 1  # already at floor
    batcher.config.max_wait_ms = 1.0
    ctrl = AdaptiveBatchController(
        batcher,
        ControllerConfig(latency_slo_ms=100.0, min_batch_size=1, min_wait_ms=1.0),
    )
    decision = ctrl.step_once()
    assert decision.max_batch_size == 1
    assert decision.max_wait_ms == 1.0


def test_clamps_at_max_batch_size():
    batcher = _seed_batcher(latencies=[10.0] * 100, qsize_target=20)
    batcher.config.max_batch_size = 32  # already at ceiling
    batcher.config.max_wait_ms = 30.0
    ctrl = AdaptiveBatchController(
        batcher,
        ControllerConfig(latency_slo_ms=100.0, max_batch_size=32, max_wait_ms=30.0),
    )
    decision = ctrl.step_once()
    assert decision.max_batch_size == 32
    assert decision.max_wait_ms == 30.0


def test_no_oscillation_under_steady_load():
    """Steady latency at 50% SLO with empty queue -> 10 holds in a row."""
    batcher = _seed_batcher(latencies=[50.0] * 100, qsize_target=0)
    ctrl = AdaptiveBatchController(batcher, ControllerConfig(latency_slo_ms=100.0))
    actions = [ctrl.step_once().action for _ in range(10)]
    assert actions == ["hold"] * 10
    # And the batcher knobs never moved.
    assert batcher.config.max_batch_size == 8
    assert batcher.config.max_wait_ms == 10.0


def test_percentile_helper():
    assert _percentile([], 50.0) is None
    assert _percentile([10.0], 95.0) == 10.0
    assert _percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50.0) == pytest.approx(3.0)


def test_recovery_after_burst():
    """Latency briefly exceeds SLO (1 shrink), then drops -> next tick holds.

    This is the failure mode we explicitly want to avoid: yo-yoing
    between shrink/grow on noisy windows.
    """
    batcher = _seed_batcher(latencies=[200.0] * 100, qsize_target=4)
    ctrl = AdaptiveBatchController(batcher, ControllerConfig(latency_slo_ms=100.0))

    d1 = ctrl.step_once()  # shrink
    assert d1.action == "shrink"
    assert batcher.config.max_batch_size == 6

    # Latency drops back to ok range, queue empties.
    batcher._recent_latencies = deque([60.0] * 100, maxlen=2048)
    batcher._queue.n = 0  # type: ignore[attr-defined]

    d2 = ctrl.step_once()
    # 60ms < 100ms SLO but qsize=0 < max_batch_size/2 -> hold, not grow.
    assert d2.action == "hold"
    assert batcher.config.max_batch_size == 6  # didn't bounce back up
