"""LRU cache for inference predictions.

Key = sha1(model_id || precision || input_text). Value = Prediction.
Thread-safe via a single lock wrapping an OrderedDict; reads also
update recency. Sized by entry count, not bytes — every entry is two
small fields.

The cache is checked at the route layer *before* the request enters
the batcher queue, so cache hits never wait for a batch to fill.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from inferbench.engine.model_runner import Prediction


def cache_key(model_id: str, precision: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(precision.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


@dataclass
class CacheStats:
    hits: int
    misses: int
    size: int
    capacity: int

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class PredictionCache:
    def __init__(self, capacity: int = 4096) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._store: OrderedDict[str, Prediction] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Prediction]:
        with self._lock:
            value = self._store.get(key)
            if value is None:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Prediction) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                size=len(self._store),
                capacity=self._capacity,
            )
