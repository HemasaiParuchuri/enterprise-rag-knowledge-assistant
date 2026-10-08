"""Text chunking for RAG ingestion.

Recursive-style splitting: split on paragraph / sentence boundaries first,
then hard-split anything still too long. Overlap keeps context across
boundaries — a detail interviewers regularly ask about.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    start_char: int


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, source: str = "document", chunk_size: int = 800,
               overlap: int = 120) -> list[Chunk]:
    """Split `text` into overlapping chunks of roughly `chunk_size` chars."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    # Normalise whitespace, keep paragraph structure as candidate units
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            units.extend(_split_sentences(para))

    chunks: list[Chunk] = []
    current, cursor = "", 0
    for unit in units:
        # Hard-split units that are still too long (e.g. no sentence breaks)
        while len(unit) > chunk_size:
            head, unit = unit[:chunk_size], unit[chunk_size - overlap:]
            chunks.append(Chunk(len(chunks), head.strip(), source, cursor))
            cursor += len(head) - overlap
        if current and len(current) + len(unit) + 1 > chunk_size:
            chunks.append(Chunk(len(chunks), current.strip(), source, cursor))
            tail = current[-overlap:] if overlap else ""
            cursor += len(current) - len(tail)
            current = (tail + " " + unit).strip()
        else:
            current = (current + "\n" + unit).strip() if current else unit
    if current.strip():
        chunks.append(Chunk(len(chunks), current.strip(), source, cursor))
    return chunks
