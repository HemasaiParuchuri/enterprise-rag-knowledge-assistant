"""Embeddings with two backends.

Default (zero-download, always works): TF-IDF implemented in pure Python.
This is the honest baseline — deterministic, fast, and a strong lexical
retriever for enterprise documents full of exact terms / acronyms.

Optional upgrade: Sentence Transformers (`USE_SBERT=1`, install
requirements-optional.txt). Same interface, semantic embeddings + FAISS
if installed. The engine picks the backend at runtime and reports which
one is active via /health, so the demo never breaks on a fresh clone.
"""
from __future__ import annotations

import math
import os
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfEmbedder:
    name = "tfidf"

    def __init__(self) -> None:
        self.idf: dict[str, float] = {}
        self.vocab: dict[str, int] = {}

    def fit(self, documents: list[str]) -> None:
        df: Counter[str] = Counter()
        for doc in documents:
            df.update(set(tokenize(doc)))
        n = max(len(documents), 1)
        self.vocab = {term: i for i, term in enumerate(sorted(df))}
        self.idf = {t: math.log((1 + n) / (1 + c)) + 1.0 for t, c in df.items()}

    def encode(self, text: str) -> dict[int, float]:
        counts = Counter(t for t in tokenize(text) if t in self.vocab)
        if not counts:
            return {}
        vec = {self.vocab[t]: c * self.idf[t] for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {i: v / norm for i, v in vec.items()}

    @staticmethod
    def cosine(a: dict[int, float], b: dict[int, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(i, 0.0) for i, v in a.items())


class SbertEmbedder:  # pragma: no cover - needs optional heavy deps
    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def fit(self, documents: list[str]) -> None:
        pass

    def encode(self, text: str):
        import numpy as np
        return np.asarray(self._model.encode(text, normalize_embeddings=True))

    @staticmethod
    def cosine(a, b) -> float:
        return float(a @ b)


def get_embedder():
    if os.getenv("USE_SBERT", "").lower() in {"1", "true", "yes"}:
        try:
            return SbertEmbedder()
        except Exception:
            pass  # fall back silently so a fresh clone always runs
    return TfidfEmbedder()
