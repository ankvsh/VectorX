"""Thin client for a local Ollama server (https://ollama.com)."""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when Ollama is unreachable or returns an unusable response."""


class OllamaClient:
    """Wraps the two Ollama endpoints this project needs.

    Setup::

        ollama pull nomic-embed-text
        ollama pull llama3.2
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 11434,
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.2",
    ) -> None:
        self._base_url = f"http://{host}:{port}"
        self.embed_model = embed_model
        self.gen_model = gen_model
        self._session = requests.Session()

    def is_available(self) -> bool:
        try:
            return self._session.get(f"{self._base_url}/api/tags", timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for ``text``."""
        data = self._post(
            "/api/embeddings",
            {"model": self.embed_model, "prompt": text},
            timeout=(3, 30),
        )
        embedding = data.get("embedding")
        if not embedding:
            raise OllamaError(
                f"Ollama returned no embedding. Is '{self.embed_model}' pulled? "
                f"Run: ollama pull {self.embed_model}"
            )
        return embedding

    def generate(self, prompt: str) -> str:
        """Return a completion for ``prompt`` (non-streaming)."""
        data = self._post(
            "/api/generate",
            {"model": self.gen_model, "prompt": prompt, "stream": False},
            timeout=(3, 180),  # local LLMs can be slow on CPU
        )
        return data.get("response", "")

    def _post(self, path: str, payload: dict, timeout: tuple[int, int]) -> dict:
        try:
            response = self._session.post(f"{self._base_url}{path}", json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            logger.warning("Ollama request to %s failed: %s", path, exc)
            raise OllamaError(
                "Ollama unavailable. Install it from https://ollama.com and run: ollama serve"
            ) from exc
        except ValueError as exc:
            raise OllamaError("Ollama returned invalid JSON") from exc
