"""Simple in-memory caches used by retrieval code."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any


class LRUCache:
    """Simple LRU Cache implementation using OrderedDict.

    This cache stores key-value pairs with a fixed capacity.
    When the capacity is exceeded, the least recently accessed item
    is evicted to make room for the new item.
    """

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self._cache: OrderedDict[str, Any] = OrderedDict()

    def get(self, key: str) -> Any | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key: str, value: Any) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()
