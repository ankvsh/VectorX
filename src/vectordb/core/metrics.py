"""Distance metrics. Smaller values always mean "more similar"."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DistanceFn = Callable[[np.ndarray, np.ndarray], float]

_EPS = 1e-9


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance."""
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def manhattan(a: np.ndarray, b: np.ndarray) -> float:
    """L1 distance."""
    return float(np.abs(a - b).sum())


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance, ``1 - cos(theta)``. Zero vectors are maximally distant."""
    norm_a = float(np.dot(a, a))
    norm_b = float(np.dot(b, b))
    if norm_a < _EPS or norm_b < _EPS:
        return 1.0
    return 1.0 - float(np.dot(a, b)) / (np.sqrt(norm_a) * np.sqrt(norm_b))


METRICS: dict[str, DistanceFn] = {
    "cosine": cosine,
    "euclidean": euclidean,
    "manhattan": manhattan,
}


def get_distance_fn(name: str) -> DistanceFn:
    """Look up a metric by name, raising ``ValueError`` for unknown names."""
    try:
        return METRICS[name]
    except KeyError:
        raise ValueError(f"unknown metric {name!r}; expected one of {sorted(METRICS)}") from None
