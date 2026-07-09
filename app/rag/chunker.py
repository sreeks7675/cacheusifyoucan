"""
Splits raw text into overlapping chunks before embedding.

Character-based (not token-based) on purpose: it keeps this module
dependency-free (no tokenizer download needed) and is precise enough
for a hackathon-scale knowledge base. Swap for a tokenizer-aware
chunker later if you need tighter control over LLM context windows.
"""

from typing import List

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Splits `text` into overlapping chunks, breaking on sentence/paragraph
    boundaries where possible so chunks don't cut words or sentences
    mid-way.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            # Prefer breaking at a sentence boundary; fall back to a
            # space; otherwise hard-cut at chunk_size.
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary <= start:
                boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = max(end - overlap, start + 1)  # guarantees forward progress

    return chunks
