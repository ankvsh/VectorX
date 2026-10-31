"""Common interface implemented by every index."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from vectordb.core import DistanceFn, Neighbor, VectorItem


class VectorIndex(ABC):
    """A k-nearest-neighbour index over :class:`VectorItem` objects."""

    @abstractmethod
    def insert(self, item: VectorItem) -> None:
        """Add ``item`` to the index."""

    @abstractmethod
    def remove(self, item_id: int) -> None:
        """Remove the item with the given id. Unknown ids are ignored."""

    @abstractmethod
    def knn(self, query: np.ndarray, k: int, dist: DistanceFn) -> list[Neighbor]:
        """Return up to ``k`` nearest neighbours as ``(distance, id)``, closest first."""

    @abstractmethod
    def __len__(self) -> int: ...
