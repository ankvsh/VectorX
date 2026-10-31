"""Health and metadata endpoints."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from vectordb.core import METRICS
from vectordb.database.vector_db import ALGORITHMS

bp = Blueprint("system", __name__)


@bp.get("/status")
def status():
    s = current_app.extensions["services"]
    return jsonify(
        ollamaAvailable=s.ollama.is_available(),
        embedModel=s.ollama.embed_model,
        genModel=s.ollama.gen_model,
        docCount=len(s.doc_db),
        docDims=s.doc_db.dims,
        demoDims=s.vector_db.dims,
        demoCount=len(s.vector_db),
    )


@bp.get("/stats")
def stats():
    db = current_app.extensions["services"].vector_db
    return jsonify(
        count=len(db),
        dims=db.dims,
        algorithms=list(ALGORITHMS),
        metrics=sorted(METRICS),
    )
