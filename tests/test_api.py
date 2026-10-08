from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["chunks"] >= 1  # sample docs indexed on startup


def test_ask_returns_citations():
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "How many vacation days do employees get?", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["citations"]
    assert body["citations"][0]["source"] == "company_handbook.txt"


def test_ingest_text_then_ask():
    with TestClient(app) as c:
        r = c.post("/ingest-text", json={
            "text": "The support team answers customer tickets within four business hours every weekday.",
            "source": "support.txt"})
        assert r.status_code == 200 and r.json()["chunks_added"] >= 1
        r = c.post("/ask", json={"question": "How quickly does support answer tickets?"})
    assert r.status_code == 200
