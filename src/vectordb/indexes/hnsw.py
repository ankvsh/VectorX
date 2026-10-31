"""HNSW: Hierarchical Navigable Small World graph (Malkov & Yashunin, 2016).

Every node is assigned a random maximum layer drawn from an exponential
distribution. Layer 0 contains all nodes; each higher layer is exponentially
sparser. A query greedily descends from the sparsest layer to find a good entry
point, then runs a beam search of width ``ef`` on layer 0.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

import numpy as np

from vectordb.core import DistanceFn, Neighbor, VectorItem, cosine
from vectordb.indexes.base import VectorIndex


@dataclass(slots=True)
class _Node:
    item: VectorItem
    max_layer: int
    neighbors: list[list[int]]  # neighbors[layer] -> ids


@dataclass(slots=True)
class GraphInfo:
    """Snapshot of the graph structure, used by the visualiser."""

    top_layer: int
    node_count: int
    nodes_per_layer: list[int] = field(default_factory=list)
    edges_per_layer: list[int] = field(default_factory=list)
    nodes: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)


class HNSWIndex(VectorIndex):
    """
    Args:
        m: Max connections per node on layers > 0 (layer 0 allows ``2 * m``).
        ef_construction: Beam width used while inserting.
        build_distance: Metric used to wire the graph. Queries may use another
            metric, but recall is best when it matches.
        seed: RNG seed so graph construction is reproducible.
    """

    def __init__(
        self,
        m: int = 16,
        ef_construction: int = 200,
        build_distance: DistanceFn = cosine,
        seed: int = 42,
    ) -> None:
        self._m = m
        self._m0 = 2 * m
        self._ef_construction = ef_construction
        self._build_dist = build_distance
        self._level_mult = 1.0 / math.log(m)
        self._rng = random.Random(seed)

        self._nodes: dict[int, _Node] = {}
        self._entry_point: int | None = None
        self._top_layer = -1

    # ------------------------------------------------------------------ insert

    def insert(self, item: VectorItem) -> None:
        level = self._random_level()
        node = _Node(item, level, [[] for _ in range(level + 1)])
        self._nodes[item.id] = node

        if self._entry_point is None:
            self._entry_point, self._top_layer = item.id, level
            return

        query = item.embedding
        entry = self._entry_point

        # Phase 1: greedy descent through layers above the node's own level.
        for layer in range(self._top_layer, level, -1):
            entry = self._greedy_step(query, entry, layer, self._build_dist)

        # Phase 2: beam search and bidirectional linking on shared layers.
        for layer in range(min(self._top_layer, level), -1, -1):
            candidates = self._search_layer(
                query, entry, self._ef_construction, layer, self._build_dist
            )
            max_conn = self._m0 if layer == 0 else self._m
            selected = [nid for _, nid in candidates[:max_conn]]
            node.neighbors[layer] = selected

            for nid in selected:
                self._link(nid, item.id, layer, max_conn)

            if candidates:
                entry = candidates[0][1]

        if level > self._top_layer:
            self._top_layer, self._entry_point = level, item.id

    def _link(self, src: int, dst: int, layer: int, max_conn: int) -> None:
        """Add ``dst`` to ``src``'s neighbour list, pruning to the closest ``max_conn``."""
        node = self._nodes[src]
        while len(node.neighbors) <= layer:
            node.neighbors.append([])
        conn = node.neighbors[layer]
        conn.append(dst)
        if len(conn) > max_conn:
            base = node.item.embedding
            scored = sorted(
                (self._build_dist(base, self._nodes[c].item.embedding), c)
                for c in conn
                if c in self._nodes
            )
            node.neighbors[layer] = [c for _, c in scored[:max_conn]]

    # ------------------------------------------------------------------ search

    def knn(self, query: np.ndarray, k: int, dist: DistanceFn, ef: int = 50) -> list[Neighbor]:
        if self._entry_point is None:
            return []
        entry = self._entry_point
        for layer in range(self._top_layer, 0, -1):
            entry = self._greedy_step(query, entry, layer, dist)
        return self._search_layer(query, entry, max(ef, k), 0, dist)[:k]

    def _greedy_step(
        self, query: np.ndarray, entry: int, layer: int, dist: DistanceFn
    ) -> int:
        if layer >= len(self._nodes[entry].neighbors):
            return entry
        best = self._search_layer(query, entry, 1, layer, dist)
        return best[0][1] if best else entry

    def _search_layer(
        self, query: np.ndarray, entry: int, ef: int, layer: int, dist: DistanceFn
    ) -> list[Neighbor]:
        """Beam search on a single layer. Returns up to ``ef`` hits, closest first."""
        d0 = dist(query, self._nodes[entry].item.embedding)
        visited = {entry}
        candidates: list[Neighbor] = [(d0, entry)]  # min-heap by distance
        found: list[Neighbor] = [(-d0, entry)]  # max-heap via negated distance

        while candidates:
            cand_dist, cand_id = heapq.heappop(candidates)
            if len(found) >= ef and cand_dist > -found[0][0]:
                break
            node = self._nodes[cand_id]
            if layer >= len(node.neighbors):
                continue
            for nid in node.neighbors[layer]:
                if nid in visited or nid not in self._nodes:
                    continue
                visited.add(nid)
                nd = dist(query, self._nodes[nid].item.embedding)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-d, i) for d, i in found)

    # ------------------------------------------------------------------ remove

    def remove(self, item_id: int) -> None:
        if item_id not in self._nodes:
            return
        del self._nodes[item_id]
        for node in self._nodes.values():
            for layer in node.neighbors:
                if item_id in layer:
                    layer.remove(item_id)

        if self._entry_point == item_id:
            self._reselect_entry_point()

    def _reselect_entry_point(self) -> None:
        """Pick the highest-layer survivor so the top layer stays consistent."""
        if not self._nodes:
            self._entry_point, self._top_layer = None, -1
            return
        entry = max(self._nodes.values(), key=lambda n: n.max_layer)
        self._entry_point, self._top_layer = entry.item.id, entry.max_layer

    # ------------------------------------------------------------ introspection

    def info(self) -> GraphInfo:
        layers = max(self._top_layer + 1, 1)
        info = GraphInfo(
            top_layer=self._top_layer,
            node_count=len(self._nodes),
            nodes_per_layer=[0] * layers,
            edges_per_layer=[0] * layers,
        )
        for node_id, node in self._nodes.items():
            info.nodes.append(
                {
                    "id": node_id,
                    "metadata": node.item.metadata,
                    "category": node.item.category,
                    "maxLyr": node.max_layer,
                }
            )
            for layer in range(min(node.max_layer, layers - 1) + 1):
                info.nodes_per_layer[layer] += 1
                if layer >= len(node.neighbors):
                    continue
                for nid in node.neighbors[layer]:
                    if node_id < nid:  # count each undirected edge once
                        info.edges_per_layer[layer] += 1
                        info.edges.append({"src": node_id, "dst": nid, "lyr": layer})
        return info

    def _random_level(self) -> int:
        # 1 - random() lies in (0, 1], so log() never sees zero.
        return int(-math.log(1.0 - self._rng.random()) * self._level_mult)

    def __len__(self) -> int:
        return len(self._nodes)
