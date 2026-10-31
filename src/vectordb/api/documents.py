"""Document storage and the RAG (retrieve + generate) pipeline."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify

from vectordb.api.errors import ApiError, int_arg, json_body
from vectordb.services import OllamaError, chunk_text

bp = Blueprint("documents", __name__, url_prefix="/doc")

_PREVIEW_CHARS = 120

_PROMPT_TEMPLATE = (
    "You are a helpful assistant. Answer the user's question directly. "
    "Use the provided context if it contains relevant information. "
    "If it doesn't, just use your own general knowledge. "
    "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
    "'the context doesn't mention'. Just answer the question naturally.\n\n"
    "Context:\n{context}"
    "Question: {question}\n\n"
    "Answer:"
)


def _services():
    return current_app.extensions["services"]


def _require_str(body: dict, key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"need {key}")
    return value


def _embed_question(body: dict):
    """Validate a question payload, embed it and retrieve the nearest chunks."""
    services = _services()
    question = _require_str(body, "question")
    k = int_arg(body, "k", 3, maximum=50)
    try:
        query = services.ollama.embed(question)
    except OllamaError as exc:
        raise ApiError(str(exc), 503) from None
    hits = services.doc_db.search(query, k, services.settings.max_retrieval_distance)
    return question, hits


@bp.post("/insert")
def insert():
    services = _services()
    body = json_body()
    title = _require_str(body, "title")
    text = _require_str(body, "text")

    chunks = chunk_text(text, services.settings.chunk_words, services.settings.chunk_overlap)

    # Embed everything first so a mid-document failure never leaves partial inserts.
    try:
        embeddings = [services.ollama.embed(chunk) for chunk in chunks]
    except OllamaError as exc:
        raise ApiError(
            f"{exc}. Then run: ollama pull {services.ollama.embed_model} "
            f"&& ollama pull {services.ollama.gen_model}",
            503,
        ) from None

    ids = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True), start=1):
        chunk_title = f"{title} [{i}/{len(chunks)}]" if len(chunks) > 1 else title
        try:
            ids.append(services.doc_db.insert(chunk_title, chunk, embedding))
        except ValueError as exc:
            raise ApiError(str(exc)) from None

    return jsonify(ids=ids, chunks=len(chunks), dims=services.doc_db.dims)


@bp.delete("/delete/<int:doc_id>")
def delete(doc_id: int):
    return jsonify(ok=_services().doc_db.remove(doc_id))


@bp.get("/list")
def list_documents():
    docs = []
    for doc in _services().doc_db.all():
        preview = doc.text[:_PREVIEW_CHARS]
        if len(doc.text) > _PREVIEW_CHARS:
            preview += "…"
        docs.append(
            {
                "id": doc.id,
                "title": doc.title,
                "preview": preview,
                "words": len(doc.text.split()),
            }
        )
    return jsonify(docs)


@bp.post("/search")
def search():
    """Retrieval only (no generation); used by the UI to highlight matches."""
    _, hits = _embed_question(json_body())
    return jsonify(
        contexts=[
            {"id": doc.id, "title": doc.title, "distance": round(dist, 4)} for dist, doc in hits
        ]
    )


@bp.post("/ask")
def ask():
    """Full RAG pipeline: embed question -> retrieve chunks -> generate answer."""
    services = _services()
    question, hits = _embed_question(json_body())

    context = "".join(
        f"[{i}] {doc.title}:\n{doc.text}\n\n" for i, (_, doc) in enumerate(hits, start=1)
    )
    prompt = _PROMPT_TEMPLATE.format(context=context, question=question)

    try:
        answer = services.ollama.generate(prompt)
    except OllamaError as exc:
        raise ApiError(str(exc), 503) from None

    return jsonify(
        answer=answer,
        model=services.ollama.gen_model,
        contexts=[
            {"id": doc.id, "title": doc.title, "text": doc.text, "distance": round(dist, 4)}
            for dist, doc in hits
        ],
        docCount=len(services.doc_db),
    )
