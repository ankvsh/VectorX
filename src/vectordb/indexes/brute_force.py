"""Exact linear-scan search. O(N * d) per query; the correctness baseline."""

from __future__ import annotations

import heapq

import numpy as np

from vectordb.core import DistanceFn, Neighbor, VectorItem
from vectordb.indexes.base import VectorIndex


class BruteForceIndex(VectorIndex):
    def __init__(self) -> None:
        self._items: dict[int, VectorItem] = {}

    def insert(self, item: VectorItem) -> None:
        self._items[item.id] = item

    def remove(self, item_id: int) -> None:
        self._items.pop(item_id, None)

    def knn(self, query: np.ndarray, k: int, dist: DistanceFn) -> list[Neighbor]:
        scored = ((dist(query, item.embedding), item.id) for item in self._items.values())
        return heapq.nsmallest(k, scored)

    def __len__(self) -> int:
        return len(self._items)
