"""Endpoints for the 16-D demo vector index."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from vectordb.api.errors import ApiError, int_arg, json_body, parse_vector_arg
from vectordb.core import METRICS, VectorItem
from vectordb.database.vector_db import ALGORITHMS

bp = Blueprint("vectors", __name__)


def _db():
    return current_app.extensions["services"].vector_db


def _rounded(vector) -> list[float]:
    return [round(float(x), 4) for x in vector]


def _item_json(item: VectorItem) -> dict:
    return {
        "id": item.id,
        "metadata": item.metadata,
        "category": item.category,
        "embedding": _rounded(item.embedding),
    }


@bp.get("/search")
def search():
    db = _db()
    query = parse_vector_arg(request.args.get("v"), db.dims)
    k = int_arg(request.args, "k", 5)
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"
    if metric not in METRICS:
        raise ApiError(f"unknown metric '{metric}'")
    if algo not in ALGORITHMS:
        raise ApiError(f"unknown algo '{algo}'")

    result = db.search(query, k, metric, algo)
    return jsonify(
        results=[{**_item_json(h.item), "distance": round(h.distance, 6)} for h in result.hits],
        latencyUs=result.latency_us,
        algo=result.algo,
        metric=result.metric,
    )


@bp.post("/insert")
def insert():
    db = _db()
    body = json_body()
    metadata = body.get("metadata")
    embedding = body.get("embedding")
    category = body.get("category") or ""
    if not isinstance(metadata, str) or not metadata:
        raise ApiError("invalid body: 'metadata' is required")
    if not isinstance(embedding, list) or len(embedding) != db.dims:
        raise ApiError(f"invalid body: 'embedding' must be a list of {db.dims} numbers")
    try:
        item_id = db.insert(metadata, str(category), [float(x) for x in embedding])
    except (TypeError, ValueError) as exc:
        raise ApiError(f"invalid body: {exc}") from None
    return jsonify(id=item_id)


@bp.delete("/delete/<int:item_id>")
def delete(item_id: int):
    return jsonify(ok=_db().remove(item_id))


@bp.get("/items")
def items():
    return jsonify([_item_json(item) for item in _db().all()])


@bp.get("/benchmark")
def benchmark():
    db = _db()
    query = parse_vector_arg(request.args.get("v"), db.dims)
    k = int_arg(request.args, "k", 5)
    metric = request.args.get("metric") or "cosine"
    if metric not in METRICS:
        raise ApiError(f"unknown metric '{metric}'")

    result = db.benchmark(query, k, metric)
    return jsonify(
        bruteforceUs=result.bruteforce_us,
        kdtreeUs=result.kdtree_us,
        hnswUs=result.hnsw_us,
        itemCount=result.item_count,
    )


@bp.get("/hnsw-info")
def hnsw_info():
    info = _db().hnsw_info()
    return jsonify(
        topLayer=info.top_layer,
        nodeCount=info.node_count,
        nodesPerLayer=info.nodes_per_layer,
        edgesPerLayer=info.edges_per_layer,
        nodes=info.nodes,
        edges=info.edges,
    )
