"""The demo vector database: one dataset, three interchangeable indexes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from vectordb.core import Neighbor, VectorItem, cosine, get_distance_fn
from vectordb.indexes import BruteForceIndex, GraphInfo, HNSWIndex, KDTreeIndex

ALGORITHMS = ("bruteforce", "kdtree", "hnsw")


@dataclass(slots=True)
class Hit:
    item: VectorItem
    distance: float


@dataclass(slots=True)
class SearchResult:
    hits: list[Hit]
    latency_us: int
    algo: str
    metric: str


@dataclass(slots=True)
class BenchmarkResult:
    bruteforce_us: int
    kdtree_us: int
    hnsw_us: int
    item_count: int


class VectorDB:
    """Thread-safe store that keeps every index in sync."""

    def __init__(self, dims: int, hnsw_m: int = 16, hnsw_ef_construction: int = 200,
                 hnsw_ef_search: int = 50) -> None:
        self.dims = dims
        self._ef_search = hnsw_ef_search
        self._store: dict[int, VectorItem] = {}
        self._brute = BruteForceIndex()
        self._kdtree = KDTreeIndex(dims)
        self._hnsw = HNSWIndex(m=hnsw_m, ef_construction=hnsw_ef_construction,
                               build_distance=cosine)
        self._lock = threading.RLock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, embedding: Sequence[float]) -> int:
        vector = np.asarray(embedding, dtype=np.float32)
        if vector.shape != (self.dims,):
            raise ValueError(f"expected a {self.dims}D vector, got shape {vector.shape}")

        with self._lock:
            item = VectorItem(self._next_id, metadata, category, vector)
            self._next_id += 1
            self._store[item.id] = item
            self._brute.insert(item)
            self._kdtree.insert(item)
            self._hnsw.insert(item)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self._store:
                return False
            del self._store[item_id]
            self._brute.remove(item_id)
            self._kdtree.remove(item_id)
            self._hnsw.remove(item_id)
            return True

    def search(self, query: Sequence[float], k: int, metric: str, algo: str) -> SearchResult:
        if algo not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {algo!r}; expected one of {list(ALGORITHMS)}")
        q = np.asarray(query, dtype=np.float32)
        runners = self._runners(q, k, metric)

        with self._lock:
            start = time.perf_counter()
            raw = runners[algo]()
            latency_us = int((time.perf_counter() - start) * 1e6)
            hits = [Hit(self._store[i], d) for d, i in raw if i in self._store]
        return SearchResult(hits, latency_us, algo, metric)

    def benchmark(self, query: Sequence[float], k: int, metric: str) -> BenchmarkResult:
        get_distance_fn(metric)  # validate early
        q = np.asarray(query, dtype=np.float32)
        runners = self._runners(q, k, metric)

        def timed(fn: Callable[[], object]) -> int:
            start = time.perf_counter()
            fn()
            return int((time.perf_counter() - start) * 1e6)

        with self._lock:
            return BenchmarkResult(
                bruteforce_us=timed(runners["bruteforce"]),
                kdtree_us=timed(runners["kdtree"]),
                hnsw_us=timed(runners["hnsw"]),
                item_count=len(self._store),
            )

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self._store.values())

    def hnsw_info(self) -> GraphInfo:
        with self._lock:
            return self._hnsw.info()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def _runners(self, q: np.ndarray, k: int, metric: str) -> dict[str, Callable[[], list[Neighbor]]]:
        dist = get_distance_fn(metric)
        return {
            "bruteforce": lambda: self._brute.knn(q, k, dist),
            "kdtree": lambda: self._kdtree.knn(q, k, dist),
            "hnsw": lambda: self._hnsw.knn(q, k, dist, ef=self._ef_search),
        }
