"""One-command demo without the API: indexes the sample docs and asks 3 questions.
Run: python scripts/demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.engine import RagEngine  # noqa: E402

engine = RagEngine()
for path in sorted((Path(__file__).parent.parent / "data" / "sample").glob("*.txt")):
    added = engine.ingest_text(path.read_text(encoding="utf-8"), source=path.name)
    print(f"Indexed {path.name}: {added} chunks")

for q in ["How many vacation days do employees get?",
          "What is retrieval augmented generation?",
          "How much of the medical premium does the company cover?"]:
    result = engine.ask(q)
    print(f"\nQ: {q}\nA: {result['answer'][:220]}...")
    print("Sources:", [(c["source"], c["score"]) for c in result["citations"]])
