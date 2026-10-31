"""Split long text into overlapping word windows for embedding."""

from __future__ import annotations


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_words`` words.

    Consecutive chunks share ``overlap_words`` words so that sentences cut at a
    boundary still appear intact in at least one chunk. Text that already fits
    in a single chunk is returned unchanged.
    """
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap_words < chunk_words:
        raise ValueError("overlap_words must be in [0, chunk_words)")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    step = chunk_words - overlap_words
    chunks: list[str] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks
