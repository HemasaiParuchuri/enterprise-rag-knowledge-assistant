"""FastAPI service for the Enterprise RAG Knowledge Assistant.

Run:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
On startup it indexes data/sample/ so /ask works immediately.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.engine import RagEngine

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"

engine = RagEngine()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if SAMPLE_DIR.exists():
        for path in sorted(SAMPLE_DIR.glob("*.txt")):
            engine.ingest_text(path.read_text(encoding="utf-8"), source=path.name)
    yield


app = FastAPI(title="Enterprise RAG Knowledge Assistant", version="1.0.0",
              description="Upload documents, ask questions, get cited answers.",
              lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How many vacation days do employees get?"])
    top_k: int = Field(3, ge=1, le=10)


class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=20)
    source: str = "pasted-text"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", **engine.stats()}


@app.get("/stats")
def stats() -> dict:
    return engine.stats()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    return engine.ask(req.question, top_k=req.top_k)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    try:
        added = engine.ingest_file(file.filename or "upload", raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"source": file.filename, "chunks_added": added, **engine.stats()}


@app.post("/ingest-text")
def ingest_text(req: IngestTextRequest) -> dict:
    added = engine.ingest_text(req.text, source=req.source)
    return {"source": req.source, "chunks_added": added, **engine.stats()}
