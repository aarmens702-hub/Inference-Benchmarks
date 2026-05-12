"""Reusable inference path shared by HTTP routes and the Gradio demo.

`run_inference` takes a ModelEntry plus the inputs and returns an
InferResponse-shaped result. It applies cache lookup, fans the misses
through the batcher (or runs them directly through the runner when no
batcher is registered), and refills the cache with fresh predictions.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from inferbench.engine.batcher import BatchResult, DynamicBatcher
from inferbench.engine.cache import PredictionCache, cache_key
from inferbench.engine.model_runner import ModelRunner
from inferbench.engine.request_queue import QueueOverflowError
from inferbench.server.registry import ModelEntry
from inferbench.server.schemas import InferResponse, PredictionItem


class InferOverloaded(Exception):
    """Raised when the queue is full. Maps to HTTP 429 at the route layer."""


class InferTimedOut(Exception):
    """Raised when a request waits longer than the configured timeout. Maps to 504."""


@dataclass
class InferConfig:
    cache: PredictionCache | None
    request_timeout_ms: float


async def run_inference(
    entry: ModelEntry,
    inputs: list[str],
    config: InferConfig,
    request_id: str | None = None,
) -> InferResponse:
    runner = entry.runner
    batcher = entry.batcher
    cache = config.cache
    timeout_sec = config.request_timeout_ms / 1000.0

    rid = request_id or str(uuid.uuid4())
    t_start = time.perf_counter()

    n = len(inputs)
    predictions: list[PredictionItem | None] = [None] * n
    miss_indices: list[int] = []
    miss_texts: list[str] = []
    cache_hits_count = 0

    if cache is not None:
        for i, text in enumerate(inputs):
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
        miss_texts = list(inputs)

    batch_size = 0
    queue_wait_ms = 0.0
    inference_ms = 0.0

    if miss_texts:
        try:
            batch_results = await _submit_misses(batcher, runner, miss_texts, timeout_sec)
        except QueueOverflowError as exc:
            raise InferOverloaded("queue full") from exc
        except asyncio.TimeoutError as exc:
            raise InferTimedOut("request timed out") from exc

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
        request_id=rid,
        predictions=predictions,  # type: ignore[arg-type]
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
        inference_ms=inference_ms,
        batch_size=batch_size if miss_texts else 0,
        cache_hit=(cache_hits_count == n and n > 0),
        model_backend=runner.metadata.backend,
        model_precision=runner.metadata.precision,
    )


async def _submit_misses(
    batcher: DynamicBatcher | None,
    runner: ModelRunner,
    miss_texts: list[str],
    timeout_sec: float,
) -> list[BatchResult]:
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
