"""Workload generators for the benchmark harness.

Each workload returns aggregated stats for one measured window:
- list of e2e latencies (ms)
- list of server-reported inference_ms / queue_wait_ms
- error count
- actual measured-window duration (sec)
- backend / precision strings echoed from the server

Workloads:
- closed_loop(N): N concurrent threads, each issuing requests serially
  (each thread waits for its response before sending the next). The
  classic benchmark pattern; pressures the server with bounded
  concurrency.
- poisson(rate_rps): open-loop arrivals — interarrival times are
  exponentially distributed with mean 1/rate. Simulates real traffic
  where the server cannot back-pressure clients.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

DEFAULT_TEXTS = [
    "this movie was surprisingly good",
    "absolute waste of time",
    "the plot was clever and the acting was solid",
    "i would not recommend this to anyone",
    "an unexpected gem with great cinematography",
    "boring, predictable, and far too long",
    "a delightful watch from start to finish",
    "the dialogue felt forced and the pacing was off",
]


@dataclass
class WorkloadResult:
    latencies_ms: list[float] = field(default_factory=list)
    inference_ms: list[float] = field(default_factory=list)
    queue_wait_ms: list[float] = field(default_factory=list)
    errors: int = 0
    duration_sec: float = 0.0
    backend: str = ""
    precision: str = ""


def _post_one(client: httpx.Client, url: str, text: str) -> tuple[float, float, float, str, str] | None:
    t0 = time.perf_counter()
    try:
        resp = client.post(url, json={"inputs": [text]})
        resp.raise_for_status()
        body = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return latency_ms, body["inference_ms"], body["queue_wait_ms"], body["model_backend"], body["model_precision"]
    except httpx.HTTPError:
        return None


def _warmup(base_url: str, duration_sec: float, texts: list[str]) -> None:
    if duration_sec <= 0:
        return
    url = f"{base_url.rstrip('/')}/infer"
    end = time.perf_counter() + duration_sec
    with httpx.Client(timeout=10.0) as client:
        i = 0
        while time.perf_counter() < end:
            try:
                client.post(url, json={"inputs": [texts[i % len(texts)]]})
            except httpx.HTTPError:
                pass
            i += 1


def closed_loop(
    base_url: str,
    concurrency: int,
    duration_sec: float,
    warmup_sec: float = 0.0,
    texts: list[str] | None = None,
) -> WorkloadResult:
    texts = texts or DEFAULT_TEXTS
    url = f"{base_url.rstrip('/')}/infer"
    _warmup(base_url, warmup_sec, texts)

    result = WorkloadResult()
    lock = threading.Lock()
    start = time.perf_counter()
    end_at = start + duration_sec

    def worker(worker_idx: int) -> None:
        with httpx.Client(timeout=10.0) as client:
            i = 0
            while time.perf_counter() < end_at:
                outcome = _post_one(client, url, texts[(worker_idx * 7919 + i) % len(texts)])
                if outcome is None:
                    with lock:
                        result.errors += 1
                else:
                    lat, inf, qw, backend, precision = outcome
                    with lock:
                        result.latencies_ms.append(lat)
                        result.inference_ms.append(inf)
                        result.queue_wait_ms.append(qw)
                        result.backend = backend
                        result.precision = precision
                i += 1

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(worker, range(concurrency)))

    result.duration_sec = time.perf_counter() - start
    return result


def poisson(
    base_url: str,
    rate_rps: float,
    duration_sec: float,
    max_inflight: int = 256,
    warmup_sec: float = 0.0,
    texts: list[str] | None = None,
    seed: int = 42,
) -> WorkloadResult:
    """Open-loop Poisson arrivals: interarrival ~ Exp(rate)."""
    texts = texts or DEFAULT_TEXTS
    url = f"{base_url.rstrip('/')}/infer"
    _warmup(base_url, warmup_sec, texts)

    result = WorkloadResult()
    lock = threading.Lock()
    rng = random.Random(seed)

    start = time.perf_counter()
    end_at = start + duration_sec

    def submit(text: str) -> None:
        with httpx.Client(timeout=10.0) as client:
            outcome = _post_one(client, url, text)
            if outcome is None:
                with lock:
                    result.errors += 1
            else:
                lat, inf, qw, backend, precision = outcome
                with lock:
                    result.latencies_ms.append(lat)
                    result.inference_ms.append(inf)
                    result.queue_wait_ms.append(qw)
                    result.backend = backend
                    result.precision = precision

    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        i = 0
        next_at = start
        while next_at < end_at:
            now = time.perf_counter()
            if now < next_at:
                time.sleep(next_at - now)
            pool.submit(submit, texts[i % len(texts)])
            i += 1
            next_at += rng.expovariate(rate_rps)

    result.duration_sec = time.perf_counter() - start
    return result
