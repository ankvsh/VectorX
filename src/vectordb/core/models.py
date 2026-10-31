"""Plain data types shared across the package."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A search hit expressed as ``(distance, item_id)``. Tuples sort by distance first,
# which is exactly what the index implementations rely on.
Neighbor = tuple[float, int]


@dataclass(slots=True)
class VectorItem:
    """A single vector together with its identifying metadata."""

    id: int
    metadata: str
    category: str
    embedding: np.ndarray
