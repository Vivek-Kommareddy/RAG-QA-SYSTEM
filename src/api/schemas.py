"""Pydantic v2 request and response models for the RAG API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class UploadResponse(BaseModel):
    """Returned after successfully ingesting one or more documents."""

    doc_ids: List[str] = Field(..., description="Unique IDs assigned to each uploaded document")
    total_chunks: int = Field(..., description="Total number of chunks created across all files")


# ---------------------------------------------------------------------------
# Ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(..., min_length=1, description="Natural-language question to answer")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Override number of chunks to retrieve (uses config default if omitted)")
    rerank_enabled: Optional[bool] = Field(None, description="Override reranking toggle (uses config default if omitted)")


class AskSource(BaseModel):
    """A single source chunk cited in an answer."""

    doc_id: Optional[str] = None
    filename: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: Optional[int] = None
    content_snippet: Optional[str] = Field(None, description="First 300 characters of the source chunk")


class AskResponse(BaseModel):
    """Response returned by the /ask endpoint."""

    answer: str = Field(..., description="Generated answer grounded in the retrieved context")
    confidence_score: float = Field(..., description="Average similarity of retrieved chunks (0–1)")
    latency_ms: float = Field(..., description="End-to-end pipeline latency in milliseconds")
    sources: List[AskSource] = Field(default_factory=list, description="Source chunks used to form the answer")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentInfo(BaseModel):
    """Metadata about a single indexed document."""

    doc_id: str
    filename: str
    num_chunks: int


class DocumentsResponse(BaseModel):
    """Response returned by GET /documents."""

    documents: List[DocumentInfo]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response returned by GET /health."""

    status: str = Field(..., description="Service status (always 'ok' if reachable)")
    doc_count: int = Field(..., description="Number of documents in the registry")
    vector_store_size: int = Field(..., description="Number of vectors in the ChromaDB collection")
