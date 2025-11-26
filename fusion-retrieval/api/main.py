"""FastAPI application for Fusion RAG"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag_core.pipeline.pipeline import FusionRAGPipeline
from rag_core.utils.config import load_config, FusionRAGConfig

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(
    title="Fusion RAG API",
    description="API for Fusion Retrieval-Augmented Generation system",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[FusionRAGPipeline] = None


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str
    k: Optional[int] = None


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    query: str
    answer: str
    context: List[str]
    scores: Optional[List[float]] = None


class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""

    data_path: str


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""

    message: str
    nodes_count: int


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    try:
        config = load_config()
        pipeline = FusionRAGPipeline(config)
    except Exception as e:
        print(f"Warning: Could not initialize pipeline on startup: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fusion RAG API",
        "status": "running",
        "pipeline_initialized": pipeline is not None,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None,
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents into the pipeline.

    Args:
        request: Ingest request with data path

    Returns:
        Ingest response with status
    """
    global pipeline

    if pipeline is None:
        config = load_config()
        pipeline = FusionRAGPipeline(config)

    data_path = request.data_path
    if not Path(data_path).exists():
        raise HTTPException(status_code=404, detail=f"File not found: {data_path}")

    try:
        pipeline.ingest(data_path)
        return IngestResponse(
            message="Documents ingested successfully",
            nodes_count=len(pipeline.nodes),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting documents: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG pipeline.

    Args:
        request: Query request

    Returns:
        Query response with answer and context
    """
    if pipeline is None or pipeline.retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Pipeline not initialized. Please ingest documents first.",
        )

    try:
        result = pipeline.query(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

