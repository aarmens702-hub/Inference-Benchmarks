"""PredictionCache correctness tests."""

from __future__ import annotations

import pytest

from inferbench.engine.cache import PredictionCache, cache_key
from inferbench.engine.model_runner import Prediction


def _pred(label: str = "POSITIVE", score: float = 0.99) -> Prediction:
    return Prediction(label=label, score=score)


def test_cache_key_changes_with_text():
    a = cache_key("m", "fp32", "hello")
    b = cache_key("m", "fp32", "world")
    assert a != b


def test_cache_key_changes_with_precision():
    a = cache_key("m", "fp32", "hello")
    b = cache_key("m", "int8", "hello")
    assert a != b


def test_cache_key_changes_with_model_id():
    a = cache_key("m1", "fp32", "hello")
    b = cache_key("m2", "fp32", "hello")
    assert a != b


def test_get_returns_none_on_miss():
    cache = PredictionCache(capacity=8)
    assert cache.get("missing") is None
    assert cache.stats().misses == 1
    assert cache.stats().hits == 0


def test_put_then_get_returns_value():
    cache = PredictionCache(capacity=8)
    cache.put("k", _pred())
    got = cache.get("k")
    assert got is not None
    assert got.label == "POSITIVE"
    assert cache.stats().hits == 1


def test_lru_evicts_oldest():
    cache = PredictionCache(capacity=2)
    cache.put("a", _pred("A"))
    cache.put("b", _pred("B"))
    cache.put("c", _pred("C"))  # should evict "a"
    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_get_promotes_recency():
    cache = PredictionCache(capacity=2)
    cache.put("a", _pred("A"))
    cache.put("b", _pred("B"))
    cache.get("a")  # touch a -> b is now LRU
    cache.put("c", _pred("C"))  # evicts b
    assert cache.get("a") is not None
    assert cache.get("b") is None


def test_put_overwrites_and_promotes():
    cache = PredictionCache(capacity=2)
    cache.put("a", _pred("A1"))
    cache.put("b", _pred("B"))
    cache.put("a", _pred("A2"))  # touch a
    cache.put("c", _pred("C"))  # evicts b, not a
    assert cache.get("a").label == "A2"
    assert cache.get("b") is None


def test_hit_ratio_is_correct():
    cache = PredictionCache(capacity=8)
    cache.put("k", _pred())
    cache.get("k")  # hit
    cache.get("k")  # hit
    cache.get("missing")  # miss
    stats = cache.stats()
    assert stats.hits == 2
    assert stats.misses == 1
    assert stats.hit_ratio == pytest.approx(2 / 3)


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        PredictionCache(capacity=0)
