"""SLO-driven adaptive batching controller (saturation-aware).

Goal: maximize throughput while keeping observed p95 latency under
`latency_slo_ms`. The controller wakes every `adjust_interval_sec`,
reads the batcher's recent per-request latencies, and steps the
batcher's `max_batch_size` and `max_wait_ms`. Mutating one knob per
tick (rather than multiplicatively rescaling) avoids oscillation.

Decision rule:

    if p95 > slo:
        if qsize > saturation_qsize_factor * max_batch_size:
            # Tail latency is queue depth, not batch wait. Smaller
            # batches make the drain rate worse and the queue
            # grows faster, so:
            grow batch (more drain), keep wait
        else:
            # Tail latency is from waiting for a batch to fill at
            # low concurrency:
            shrink wait, keep batch
    elif p95 < grow_under_ratio * slo and qsize > max_batch_size / 2:
        grow batch + grow wait — we have slack, harvest throughput
    else:
        hold

The saturation branch is the W7 follow-up — bench(C-adaptive-v1)
showed that without it, the controller responds to overload by
shrinking the very thing (batch size) that was carrying drain
throughput, which makes the regression worse.

A no-data tick (fewer than min_samples observed) holds settings.
"""

from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional

from inferbench.engine.batcher import DynamicBatcher

log = logging.getLogger(__name__)


@dataclass
class ControllerConfig:
    latency_slo_ms: float = 100.0
    adjust_interval_sec: float = 5.0
    min_batch_size: int = 1
    max_batch_size: int = 32
    min_wait_ms: float = 1.0
    max_wait_ms: float = 30.0
    # Hysteresis: only grow when observed latency is well under SLO.
    grow_under_ratio: float = 0.7
    # How much each tick can move a knob.
    batch_step: int = 2
    wait_step_ms: float = 2.0
    # Minimum samples required before any decision is made.
    min_samples: int = 32
    # Saturation detection: qsize > this factor * max_batch_size
    # implies queue-depth latency, not batch-wait latency.
    saturation_qsize_factor: float = 2.0


@dataclass
class ControllerDecision:
    tick: int
    p95_ms: Optional[float]
    samples: int
    qsize: int
    action: str  # "grow" | "shrink" | "hold" | "no-data"
    max_batch_size: int
    max_wait_ms: float


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


class AdaptiveBatchController:
    def __init__(self, batcher: DynamicBatcher, config: ControllerConfig | None = None) -> None:
        self.batcher = batcher
        self.config = config or ControllerConfig()
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._tick = 0
        self._decisions: list[ControllerDecision] = []
        self._max_decision_history = 500

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("Controller already started")
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="adaptive-controller")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    @property
    def decisions(self) -> list[ControllerDecision]:
        return list(self._decisions)

    def step_once(self) -> ControllerDecision:
        """Public single-tick decision; useful for tests and CLI replay."""
        self._tick += 1
        latencies = self.batcher.recent_latencies()
        qsize = self.batcher.qsize()
        cfg = self.config
        bcfg = self.batcher.config

        if len(latencies) < cfg.min_samples:
            decision = ControllerDecision(
                tick=self._tick,
                p95_ms=None,
                samples=len(latencies),
                qsize=qsize,
                action="no-data",
                max_batch_size=bcfg.max_batch_size,
                max_wait_ms=bcfg.max_wait_ms,
            )
            self._record(decision)
            return decision

        p95 = _percentile(latencies, 95.0) or 0.0

        action: str
        if p95 > cfg.latency_slo_ms:
            if qsize > cfg.saturation_qsize_factor * bcfg.max_batch_size:
                # Saturation — tail latency is queue depth. Larger batches
                # increase drain throughput; keep wait so we don't lose
                # the bigger batches we're now harvesting.
                new_batch = min(cfg.max_batch_size, bcfg.max_batch_size + cfg.batch_step)
                new_wait = bcfg.max_wait_ms
                action = "grow-saturation"
            else:
                # Batch-wait jitter at low load — shorten the wait so a
                # solitary request flushes faster. Keep batch in case
                # load picks up.
                new_batch = bcfg.max_batch_size
                new_wait = max(cfg.min_wait_ms, bcfg.max_wait_ms - cfg.wait_step_ms)
                action = "shrink-wait"
        elif p95 < cfg.grow_under_ratio * cfg.latency_slo_ms and qsize > bcfg.max_batch_size / 2:
            new_batch = min(cfg.max_batch_size, bcfg.max_batch_size + cfg.batch_step)
            new_wait = min(cfg.max_wait_ms, bcfg.max_wait_ms + cfg.wait_step_ms)
            action = "grow"
        else:
            new_batch = bcfg.max_batch_size
            new_wait = bcfg.max_wait_ms
            action = "hold"

        bcfg.max_batch_size = new_batch
        bcfg.max_wait_ms = new_wait

        decision = ControllerDecision(
            tick=self._tick,
            p95_ms=p95,
            samples=len(latencies),
            qsize=qsize,
            action=action,
            max_batch_size=new_batch,
            max_wait_ms=new_wait,
        )
        self._record(decision)
        return decision

    def _record(self, decision: ControllerDecision) -> None:
        self._decisions.append(decision)
        if len(self._decisions) > self._max_decision_history:
            self._decisions = self._decisions[-self._max_decision_history :]

    async def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                try:
                    await asyncio.wait_for(self._stopping.wait(), timeout=self.config.adjust_interval_sec)
                    return  # stop signaled
                except asyncio.TimeoutError:
                    pass
                self.step_once()
        except asyncio.CancelledError:
            return
