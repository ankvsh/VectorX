"""Flask application factory."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from flask import Flask, Response, jsonify, send_from_directory

from vectordb.api.errors import ApiError
from vectordb.config import Settings
from vectordb.database import DocumentDB, VectorDB
from vectordb.services import OllamaClient

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@dataclass
class Services:
    """Everything the route handlers depend on, injected so tests can swap parts."""

    settings: Settings
    vector_db: VectorDB
    doc_db: DocumentDB
    ollama: OllamaClient


def build_services(settings: Settings) -> Services:
    from vectordb.data import load_demo_data

    vector_db = VectorDB(
        settings.demo_dims,
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    load_demo_data(vector_db)
    doc_db = DocumentDB(
        hnsw_m=settings.hnsw_m,
        hnsw_ef_construction=settings.hnsw_ef_construction,
        hnsw_ef_search=settings.hnsw_ef_search,
    )
    ollama = OllamaClient(
        settings.ollama_host, settings.ollama_port, settings.embed_model, settings.gen_model
    )
    return Services(settings, vector_db, doc_db, ollama)


def create_app(services: Services | None = None, settings: Settings | None = None) -> Flask:
    """Build the Flask app. Pass ``services`` to inject custom dependencies."""
    from vectordb.api.documents import bp as documents_bp
    from vectordb.api.system import bp as system_bp
    from vectordb.api.vectors import bp as vectors_bp

    app = Flask(__name__)
    app.json.sort_keys = False
    app.extensions["services"] = services or build_services(settings or Settings())

    for blueprint in (vectors_bp, documents_bp, system_bp):
        app.register_blueprint(blueprint)

    @app.get("/")
    def index() -> Response:
        return send_from_directory(STATIC_DIR, "index.html")

    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return jsonify(error=exc.message), exc.status

    @app.errorhandler(404)
    def handle_not_found(_exc):
        return jsonify(error="not found"), 404

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return jsonify(error=exc.description), exc.code
        logger.exception("Unhandled error")
        return jsonify(error="internal server error"), 500

    return app
