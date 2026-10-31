"""K-D tree: exact search via axis-aligned space partitioning.

Works well in low dimensions (roughly <= 20). In high dimensions almost no
subtree can be pruned, so it degrades towards a linear scan.

Note: the pruning test compares the distance along a single axis against the
current k-th best distance. That bound is only strictly valid for Euclidean
distance, so results under cosine/manhattan are approximate.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable

import numpy as np

from vectordb.core import DistanceFn, Neighbor, VectorItem
from vectordb.indexes.base import VectorIndex


class _Node:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem) -> None:
        self.item = item
        self.left: _Node | None = None
        self.right: _Node | None = None


class KDTreeIndex(VectorIndex):
    def __init__(self, dims: int) -> None:
        self._dims = dims
        self._root: _Node | None = None
        self._size = 0

    def insert(self, item: VectorItem) -> None:
        node = _Node(item)
        self._size += 1
        if self._root is None:
            self._root = node
            return

        current, depth = self._root, 0
        while True:
            axis = depth % self._dims
            go_left = item.embedding[axis] < current.item.embedding[axis]
            child = current.left if go_left else current.right
            if child is None:
                if go_left:
                    current.left = node
                else:
                    current.right = node
                return
            current, depth = child, depth + 1

    def remove(self, item_id: int) -> None:
        """Removal rebuilds the tree; K-D trees have no cheap delete."""
        remaining = [item for item in self._iter_items() if item.id != item_id]
        if len(remaining) != self._size:
            self.rebuild(remaining)

    def rebuild(self, items: Iterable[VectorItem]) -> None:
        self._root, self._size = None, 0
        for item in items:
            self.insert(item)

    def knn(self, query: np.ndarray, k: int, dist: DistanceFn) -> list[Neighbor]:
        # Max-heap of the best k so far, stored as (-distance, id).
        heap: list[Neighbor] = []
        self._search(self._root, query, k, 0, dist, heap)
        return sorted((-d, i) for d, i in heap)

    def _search(
        self,
        node: _Node | None,
        query: np.ndarray,
        k: int,
        depth: int,
        dist: DistanceFn,
        heap: list[Neighbor],
    ) -> None:
        if node is None:
            return

        d = dist(query, node.item.embedding)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item.id))

        axis = depth % self._dims
        diff = float(query[axis] - node.item.embedding[axis])
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)

        self._search(closer, query, k, depth + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._search(farther, query, k, depth + 1, dist, heap)

    def _iter_items(self) -> Iterable[VectorItem]:
        stack = [self._root] if self._root else []
        while stack:
            node = stack.pop()
            yield node.item
            stack.extend(child for child in (node.left, node.right) if child)

    def __len__(self) -> int:
        return self._size
