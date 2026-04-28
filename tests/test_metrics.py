"""Percentile + throughput math correctness."""

from __future__ import annotations

import pytest

from inferbench.engine.metrics import latency_stats, percentile, throughput_stats


def test_percentile_known_values():
    xs = sorted([1.0, 2.0, 3.0, 4.0, 5.0])
    assert percentile(xs, 0.0) == 1.0
    assert percentile(xs, 100.0) == 5.0
    assert percentile(xs, 50.0) == 3.0


def test_percentile_interpolates():
    xs = [10.0, 20.0]
    assert percentile(xs, 50.0) == pytest.approx(15.0)


def test_percentile_single_value():
    assert percentile([42.0], 95.0) == 42.0


def test_percentile_rejects_empty():
    with pytest.raises(ValueError):
        percentile([], 50.0)


def test_percentile_rejects_out_of_range():
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], -1.0)
    with pytest.raises(ValueError):
        percentile([1.0, 2.0], 101.0)


def test_latency_stats_basic():
    stats = latency_stats([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    assert stats.count == 10
    assert stats.min_ms == 10.0
    assert stats.max_ms == 100.0
    assert stats.mean_ms == 55.0
    assert stats.p50_ms == pytest.approx(55.0)
    assert stats.p95_ms == pytest.approx(95.5, abs=0.1)


def test_latency_stats_rejects_empty():
    with pytest.raises(ValueError):
        latency_stats([])


def test_throughput_stats_basic():
    t = throughput_stats(requests=100, errors=2, duration_sec=10.0)
    assert t.requests == 100
    assert t.errors == 2
    assert t.requests_per_sec == 10.0
    assert t.error_rate == 0.02


def test_throughput_stats_rejects_zero_duration():
    with pytest.raises(ValueError):
        throughput_stats(requests=1, errors=0, duration_sec=0)
