"""Latency / throughput percentile math.

This is the single source of truth for percentile and throughput
calculations across the project. Every benchmark scenario must call
into here rather than recomputing percentiles inline.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass


@dataclass
class LatencyStats:
    count: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ThroughputStats:
    duration_sec: float
    requests: int
    requests_per_sec: float
    errors: int
    error_rate: float

    def to_dict(self) -> dict:
        return asdict(self)


def percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile, p in [0, 100].

    Assumes input is already sorted ascending. Caller sorts once and
    passes the same list for all percentiles to avoid re-sorting.
    """
    if not sorted_values:
        raise ValueError("Cannot compute percentile of empty list")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p}")

    if len(sorted_values) == 1:
        return sorted_values[0]

    rank = (p / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        raise ValueError("Cannot compute stats over zero samples")
    sorted_ms = sorted(latencies_ms)
    return LatencyStats(
        count=len(sorted_ms),
        mean_ms=statistics.fmean(sorted_ms),
        p50_ms=percentile(sorted_ms, 50.0),
        p90_ms=percentile(sorted_ms, 90.0),
        p95_ms=percentile(sorted_ms, 95.0),
        p99_ms=percentile(sorted_ms, 99.0),
        min_ms=sorted_ms[0],
        max_ms=sorted_ms[-1],
    )


def throughput_stats(requests: int, errors: int, duration_sec: float) -> ThroughputStats:
    """`requests` is the count of *successful* requests. `errors` is failures.
    error_rate is fraction-of-attempts that failed: errors / (requests + errors).
    """
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be > 0, got {duration_sec}")
    total_attempted = requests + errors
    return ThroughputStats(
        duration_sec=duration_sec,
        requests=requests,
        requests_per_sec=requests / duration_sec,
        errors=errors,
        error_rate=errors / total_attempted if total_attempted else 0.0,
    )
