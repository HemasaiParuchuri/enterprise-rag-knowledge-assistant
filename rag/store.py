"""In-memory vector store. Interface mirrors FAISS/Qdrant (add / search),
so swapping in a persistent store is a one-class change — see README."""
from __future__ import annotations

from .chunking import Chunk
from .embeddings import TfidfEmbedder, get_embedder


class VectorStore:
    def __init__(self, embedder=None) -> None:
        self.embedder = embedder or get_embedder()
        self.chunks: list[Chunk] = []
        self._vectors: list = []

    def add(self, chunks: list[Chunk]) -> int:
        self.chunks.extend(chunks)
        # TF-IDF vocab depends on the corpus, so re-fit then re-encode all.
        if isinstance(self.embedder, TfidfEmbedder):
            self.embedder.fit([c.text for c in self.chunks])
            self._vectors = [self.embedder.encode(c.text) for c in self.chunks]
        else:
            self._vectors.extend(self.embedder.encode(c.text) for c in chunks)
        return len(chunks)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        qvec = self.embedder.encode(query)
        scored = [
            {"chunk": chunk, "score": round(float(self.embedder.cosine(qvec, vec)), 4)}
            for chunk, vec in zip(self.chunks, self._vectors)
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return [r for r in scored[:top_k] if r["score"] > 0]

    def __len__(self) -> int:
        return len(self.chunks)

    def clear(self) -> None:
        self.chunks, self._vectors = [], []
