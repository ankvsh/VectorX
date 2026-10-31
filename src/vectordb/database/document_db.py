"""Document store: HNSW over real (high-dimensional) embeddings."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from vectordb.core import VectorItem, cosine
from vectordb.indexes import BruteForceIndex, HNSWIndex

# Below this size a linear scan is both exact and faster than graph traversal.
_BRUTE_FORCE_THRESHOLD = 10


@dataclass(slots=True)
class DocItem:
    id: int
    title: str
    text: str
    embedding: np.ndarray


class DocumentDB:
    """Stores text chunks with their embeddings and retrieves them by cosine distance."""

    def __init__(self, hnsw_m: int = 16, hnsw_ef_construction: int = 200,
                 hnsw_ef_search: int = 50) -> None:
        self._ef_search = hnsw_ef_search
        self._store: dict[int, DocItem] = {}
        self._hnsw = HNSWIndex(m=hnsw_m, ef_construction=hnsw_ef_construction,
                               build_distance=cosine)
        self._brute = BruteForceIndex()
        self._lock = threading.RLock()
        self._next_id = 1
        self.dims = 0  # set by the first insert

    def insert(self, title: str, text: str, embedding: Sequence[float]) -> int:
        vector = np.asarray(embedding, dtype=np.float32)
        with self._lock:
            if self.dims == 0:
                self.dims = vector.shape[0]
            elif vector.shape != (self.dims,):
                raise ValueError(f"expected a {self.dims}D embedding, got {vector.shape[0]}D")

            doc = DocItem(self._next_id, title, text, vector)
            self._next_id += 1
            self._store[doc.id] = doc
            index_item = VectorItem(doc.id, title, "doc", vector)
            self._hnsw.insert(index_item)
            self._brute.insert(index_item)
            return doc.id

    def search(
        self, query: Sequence[float], k: int, max_distance: float = 0.7
    ) -> list[tuple[float, DocItem]]:
        """Return up to ``k`` chunks within ``max_distance``, closest first."""
        q = np.asarray(query, dtype=np.float32)
        with self._lock:
            if not self._store:
                return []
            if len(self._store) < _BRUTE_FORCE_THRESHOLD:
                raw = self._brute.knn(q, k, cosine)
            else:
                raw = self._hnsw.knn(q, k, cosine, ef=self._ef_search)
            return [(d, self._store[i]) for d, i in raw if i in self._store and d <= max_distance]

    def remove(self, doc_id: int) -> bool:
        with self._lock:
            if doc_id not in self._store:
                return False
            del self._store[doc_id]
            self._hnsw.remove(doc_id)
            self._brute.remove(doc_id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self._store.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
