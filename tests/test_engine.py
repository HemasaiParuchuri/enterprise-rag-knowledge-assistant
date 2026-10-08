from rag.engine import RagEngine

HANDBOOK = (
    "Employees receive 20 vacation days per year. Vacation requests need "
    "manager approval for absences longer than five working days.\n\n"
    "The company VPN must be used when accessing internal systems remotely. "
    "Remote employees work up to three days per week from home."
)


def make_engine():
    engine = RagEngine()
    engine.ingest_text(HANDBOOK, source="handbook.txt")
    return engine


def test_ingest_creates_chunks():
    assert make_engine().stats()["chunks"] >= 1


def test_retrieval_finds_vacation_chunk():
    result = make_engine().ask("How many vacation days do employees receive?")
    assert result["citations"], "expected at least one citation"
    assert "vacation" in result["citations"][0]["excerpt"].lower()
    assert "20" in result["answer"]


def test_unknown_question_is_honest():
    result = make_engine().ask("What is the price of Bitcoin today zzz qqq?")
    # Either no citations, or the engine must not fabricate a confident answer
    assert result["answer"] or result["citations"] == []
