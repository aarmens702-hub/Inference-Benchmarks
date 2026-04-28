"""Async request queue used by the dynamic batcher.

W3: thin asyncio.Queue wrapper with a typed put/get and a
QueueOverflowError raised on overflow. W4: bounds the queue
(max_size > 0) so put_nowait raises immediately when full — the route
maps that to HTTP 429 instead of queueing unbounded latency.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar


class QueueOverflowError(Exception):
    """Raised when the queue is full and put_nowait would block."""


T = TypeVar("T")


@dataclass
class BatchedRequest:
    """One inference unit waiting in the queue.

    `future` resolves to (prediction, batch_size, queue_wait_ms,
    inference_ms) when the batcher processes the batch this item
    belongs to.
    """

    text: str
    submitted_at: float = field(default_factory=time.perf_counter)
    future: asyncio.Future = field(default=None)  # type: ignore[assignment]


class RequestQueue(Generic[T]):
    """Thin asyncio.Queue wrapper. Backpressure logic lands in W4."""

    def __init__(self, max_size: int = 0) -> None:
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=max_size)

    def qsize(self) -> int:
        return self._queue.qsize()

    def put_nowait(self, item: T) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise QueueOverflowError("request queue full") from exc

    async def get(self) -> T:
        return await self._queue.get()

    async def get_with_timeout(self, timeout_sec: float) -> T | None:
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout_sec)
        except asyncio.TimeoutError:
            return None
