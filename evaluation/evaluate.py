"""Retrieval evaluation: Top-k accuracy on labelled question -> expected source.
Run: python -m evaluation.evaluate
This is the script behind the evaluation numbers quoted in the README.
"""
from pathlib import Path

from rag.engine import RagEngine

CASES = [
    ("How many vacation days do employees get?", "company_handbook.txt"),
    ("How many days can employees work remotely?", "company_handbook.txt"),
    ("What percentage of the medical premium does the company cover?", "company_handbook.txt"),
    ("What is retrieval augmented generation?", "ml_glossary.txt"),
    ("What is chunking and why does overlap matter?", "ml_glossary.txt"),
    ("How is retrieval quality measured?", "ml_glossary.txt"),
]


def main(top_k: int = 3) -> None:
    engine = RagEngine()
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
    for path in sorted(sample_dir.glob("*.txt")):
        engine.ingest_text(path.read_text(encoding="utf-8"), source=path.name)
    hits = 0
    for question, expected in CASES:
        result = engine.ask(question, top_k=top_k)
        sources = [c["source"] for c in result["citations"]]
        ok = expected in sources
        hits += ok
        print(f"{'PASS' if ok else 'MISS'}  top{top_k}={sources}  Q: {question}")
    print(f"\nTop-{top_k} accuracy: {hits}/{len(CASES)} = {hits / len(CASES):.0%}")


if __name__ == "__main__":
    main()
