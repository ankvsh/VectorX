"""Application settings, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VECTORDB_{name}", default)


@dataclass(frozen=True)
class Settings:
    """Runtime configuration.

    Every field can be overridden with a ``VECTORDB_<FIELD>`` environment
    variable, e.g. ``VECTORDB_PORT=9000`` or ``VECTORDB_GEN_MODEL=llama3.2:1b``.
    """

    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8080")))
    debug: bool = field(default_factory=lambda: _env("DEBUG", "0") == "1")

    # Demo index
    demo_dims: int = 16
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 50

    # Ollama
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "127.0.0.1"))
    ollama_port: int = field(default_factory=lambda: int(_env("OLLAMA_PORT", "11434")))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "nomic-embed-text"))
    gen_model: str = field(default_factory=lambda: _env("GEN_MODEL", "llama3.2"))

    # Document chunking / retrieval
    chunk_words: int = 250
    chunk_overlap: int = 30
    max_retrieval_distance: float = 0.7
