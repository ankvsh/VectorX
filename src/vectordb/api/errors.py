"""API error type and request-parsing helpers."""

from __future__ import annotations

from typing import Any

from flask import request


class ApiError(Exception):
    """An error that is rendered to the client as ``{"error": message}``."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def json_body() -> dict[str, Any]:
    """Return the request's JSON object body, or raise a 400."""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError("request body must be a JSON object")
    return body


def int_arg(source: Any, name: str, default: int, minimum: int = 1, maximum: int = 1000) -> int:
    """Parse an optional integer from query args or a JSON body, clamped to a range."""
    raw = source.get(name)
    if raw in (None, ""):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ApiError(f"'{name}' must be an integer") from None
    return max(minimum, min(maximum, value))


def parse_vector_arg(raw: str | None, dims: int) -> list[float]:
    """Parse ``"0.1,0.2,..."`` into a list of floats of the expected length."""
    try:
        vector = [float(part) for part in (raw or "").split(",") if part.strip()]
    except ValueError:
        raise ApiError("'v' must be comma-separated numbers") from None
    if len(vector) != dims:
        raise ApiError(f"need {dims}D vector")
    return vector
