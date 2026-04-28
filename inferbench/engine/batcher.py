"""DynamicBatcher: collects requests by max_batch_size or max_wait_ms.

Algorithm (per loop iteration):
  1. Block on the queue until a request arrives -> seed `batch=[r0]`.
  2. Compute deadline = r0.submitted_at + max_wait_ms.
  3. Pull more items, each with timeout = max(0, deadline - now), until
     either the batch reaches max_batch_size or the deadline passes.
  4. Hand the batch to ModelRunner.run via asyncio.to_thread (CPU-bound,
     can't block the event loop). Resolve each request's future with its
     prediction, the actual batch size, queue_wait_ms (per-request:
     inference_start - submitted_at) and inference_ms (shared across
     the batch).

Per-request queue_wait_ms varies: the first request in a batch waits up
to max_wait_ms; the last waits ~0ms. inference_ms is shared because they
all execute in one forward pass.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from inferbench.engine.model_runner import ModelRunner, Prediction
from inferbench.engine.request_queue import BatchedRequest, RequestQueue


@dataclass(frozen=True)
class BatcherConfig:
    max_batch_size: int = 16
    max_wait_ms: float = 10.0
    queue_max_size: int = 0  # 0 = unbounded; W4 sets this


@dataclass(frozen=True)
class BatchResult:
    prediction: Prediction
    batch_size: int
    queue_wait_ms: float
    inference_ms: float


class DynamicBatcher:
    def __init__(self, runner: ModelRunner, config: BatcherConfig | None = None) -> None:
        self.runner = runner
        self.config = config or BatcherConfig()
        self._queue: RequestQueue[BatchedRequest] = RequestQueue(max_size=self.config.queue_max_size)
        self._task: asyncio.Task | None = None
        self._stopping = asyncio.Event()
        self._batches_processed = 0
        self._items_processed = 0
        self._batch_size_sum = 0

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("Batcher already started")
        self._stopping.clear()
        self._task = asyncio.create_task(self._run(), name="dynamic-batcher")

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

    def qsize(self) -> int:
        return self._queue.qsize()

    @property
    def avg_batch_size(self) -> float:
        if self._batches_processed == 0:
            return 0.0
        return self._batch_size_sum / self._batches_processed

    async def submit(self, text: str) -> BatchResult:
        loop = asyncio.get_running_loop()
        req = BatchedRequest(text=text, future=loop.create_future())
        self._queue.put_nowait(req)
        return await req.future

    async def _run(self) -> None:
        try:
            while not self._stopping.is_set():
                batch = await self._collect_batch()
                if not batch:
                    continue
                await self._process_batch(batch)
        except asyncio.CancelledError:
            return

    async def _collect_batch(self) -> list[BatchedRequest]:
        first = await self._queue.get()
        batch = [first]
        deadline = first.submitted_at + (self.config.max_wait_ms / 1000.0)

        while len(batch) < self.config.max_batch_size:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            item = await self._queue.get_with_timeout(remaining)
            if item is None:
                break
            batch.append(item)
        return batch

    async def _process_batch(self, batch: list[BatchedRequest]) -> None:
        texts = [r.text for r in batch]
        inference_start = time.perf_counter()

        try:
            run_result = await asyncio.to_thread(self.runner.run, texts)
        except Exception as exc:  # surface error to all batch members
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return

        self._batches_processed += 1
        self._items_processed += len(batch)
        self._batch_size_sum += len(batch)

        for i, req in enumerate(batch):
            queue_wait_ms = (inference_start - req.submitted_at) * 1000.0
            result = BatchResult(
                prediction=run_result.predictions[i],
                batch_size=len(batch),
                queue_wait_ms=queue_wait_ms,
                inference_ms=run_result.inference_ms,
            )
            if not req.future.done():
                req.future.set_result(result)
