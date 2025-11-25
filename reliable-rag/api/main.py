from fastapi import FastAPI
from pydantic import BaseModel

from rag_core.pipeline.pipeline import ReliableRAGPipeline

app = FastAPI(title="Reliable RAG API")
pipeline = ReliableRAGPipeline()


class Query(BaseModel):
    """Request body for a RAG query."""

    question: str


class QueryResponse(BaseModel):
    """Response payload returned by the RAG API."""

    question: str
    answer: str
    hallucination_score: str | None = None
    highlights: dict | None = None


@app.post("/query", response_model=QueryResponse)
def query_rag(payload: Query) -> QueryResponse:
    """Run the Reliable RAG pipeline for an incoming question."""
    result = pipeline.run(payload.question)
    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        hallucination_score=result["hallucination_score"],
        highlights=result["highlights"],
    )

