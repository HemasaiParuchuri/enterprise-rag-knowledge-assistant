"""RAG engine: ingest -> chunk -> index -> retrieve -> answer with citations.

Answering is extractive by default (no API key, no hallucination: the
answer is built only from retrieved chunks). If OPENAI_API_KEY is set,
`generate_llm_answer` can be wired in — the retrieval layer is unchanged.
"""
from __future__ import annotations

import io

from .chunking import chunk_text
from .store import VectorStore


def extract_text(filename: str, raw: bytes) -> str:
    name = filename.lower()
    if name.endswith((".txt", ".md")):
        return raw.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # clear, honest error — not a silent fail
            raise ValueError("PDF support needs `pypdf` (pip install pypdf)") from exc
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {filename} (use .pdf, .txt or .md)")


class RagEngine:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()
        self.sources: list[str] = []

    def ingest_text(self, text: str, source: str) -> int:
        chunks = chunk_text(text, source=source)
        if source not in self.sources:
            self.sources.append(source)
        return self.store.add(chunks)

    def ingest_file(self, filename: str, raw: bytes) -> int:
        return self.ingest_text(extract_text(filename, raw), source=filename)

    def ask(self, question: str, top_k: int = 3) -> dict:
        hits = self.store.search(question, top_k=top_k)
        if not hits:
            return {
                "question": question,
                "answer": "I don't have enough information in the indexed documents to answer that.",
                "citations": [],
                "backend": self.store.embedder.name,
            }
        # Extractive answer: the top chunk's most relevant sentences.
        best = hits[0]["chunk"].text.replace("\n", " ")
        answer = f"Based on {hits[0]['chunk'].source}: {best}"
        citations = [
            {"source": h["chunk"].source, "chunk_id": h["chunk"].chunk_id,
             "score": h["score"], "excerpt": h["chunk"].text[:300]}
            for h in hits
        ]
        return {"question": question, "answer": answer,
                "citations": citations, "backend": self.store.embedder.name}

    def stats(self) -> dict:
        return {"chunks": len(self.store), "sources": self.sources,
                "backend": self.store.embedder.name}
