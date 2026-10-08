import pytest

from rag.chunking import chunk_text


def test_short_text_is_one_chunk():
    chunks = chunk_text("Hello world. This is a test.", source="a.txt")
    assert len(chunks) == 1
    assert chunks[0].source == "a.txt"


def test_long_text_splits_with_overlap():
    text = " ".join(f"Sentence number {i} explains topic {i} in detail." for i in range(120))
    chunks = chunk_text(text, source="long.txt", chunk_size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 400 for c in chunks)


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
