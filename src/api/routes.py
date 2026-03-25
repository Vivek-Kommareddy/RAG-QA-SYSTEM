"""API route definitions.

All business-logic endpoints are registered here on an :class:`APIRouter`.
The router is mounted by :mod:`src.api.main`.

Endpoints
---------
- ``POST /upload``          – Ingest one or more documents.
- ``POST /ask``             – Answer a natural-language question.
- ``GET  /documents``       – List indexed documents.
- ``DELETE /documents/{id}`` – Remove a document from the index.
- ``GET  /health``          – Service health check.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request, UploadFile, File

from ..config import get_settings
from ..ingestion.loader import load_file
from ..ingestion.chunker import chunk_documents
from ..generation.chain import RAGResponse
from .schemas import (
    AskRequest,
    AskResponse,
    AskSource,
    DocumentInfo,
    DocumentsResponse,
    HealthResponse,
    UploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------


@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...),
) -> UploadResponse:
    """Ingest one or more documents and add them to the vector store.

    Each uploaded file is loaded, chunked, embedded, and stored in ChromaDB.
    A unique ``doc_id`` is assigned to every file and returned in the response.

    Args:
        request: FastAPI request (used to access ``app.state``).
        files: One or more uploaded files (PDF, TXT, MD, DOCX).

    Returns:
        :class:`UploadResponse` with assigned doc IDs and total chunk count.

    Raises:
        HTTPException 400: If no files are provided.
        HTTPException 422: If a file format is not supported.
    """
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be provided.")

    settings = get_settings()
    embedder = request.app.state.embedder
    vectorstore = request.app.state.vectorstore
    doc_registry: Dict[str, Dict[str, Any]] = request.app.state.doc_registry

    doc_ids: List[str] = []
    total_chunks = 0

    for upload in files:
        suffix = os.path.splitext(upload.filename or "")[1].lower() or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await upload.read())
            tmp_path = tmp.name

        # --- Load ---
        try:
            docs = load_file(tmp_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to load '%s': %s", upload.filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Failed to load file '{upload.filename}': {exc}") from exc
        finally:
            os.unlink(tmp_path)

        if not docs or all(not d.content.strip() for d in docs):
            raise HTTPException(
                status_code=422,
                detail=f"'{upload.filename}' appears to be empty or contains no extractable text.",
            )

        doc_id = str(uuid.uuid4())
        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            # Store chunk text in metadata so retrieval can surface it
            chunk.metadata["content"] = chunk.content

        texts = [c.content for c in chunks]

        # --- Embed ---
        try:
            embeddings = embedder.embed_texts(texts)
        except Exception as exc:
            logger.error("Embedding failed for '%s': %s", upload.filename, exc, exc_info=True)
            raise HTTPException(
                status_code=502,
                detail=f"Embedding failed — check your OPENAI_API_KEY is valid. Error: {exc}",
            ) from exc

        # --- Store ---
        metadatas = [c.metadata for c in chunks]
        try:
            vectorstore.add_documents(embeddings.tolist(), metadatas)
        except Exception as exc:
            logger.error("Vector store insertion failed for '%s': %s", upload.filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Vector store error: {exc}") from exc

        doc_registry[doc_id] = {
            "filename": upload.filename or "unknown",
            "num_chunks": len(chunks),
        }
        doc_ids.append(doc_id)
        total_chunks += len(chunks)
        logger.info("Ingested '%s' → doc_id=%s (%d chunks)", upload.filename, doc_id, len(chunks))

    return UploadResponse(doc_ids=doc_ids, total_chunks=total_chunks)


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """Answer a natural-language question using the RAG pipeline.

    Args:
        request: FastAPI request (used to access ``app.state``).
        body: Request body containing the question.

    Returns:
        :class:`AskResponse` with the answer, source citations, confidence
        score, and latency.
    """
    rag_response: RAGResponse = request.app.state.chain.ask(
        body.question,
        top_k=body.top_k,
        rerank_enabled=body.rerank_enabled,
    )

    sources: List[AskSource] = []
    for meta in rag_response.sources:
        sources.append(
            AskSource(
                doc_id=meta.get("doc_id"),
                filename=meta.get("filename"),
                page_number=meta.get("page_number"),  # type: ignore[arg-type]
                chunk_index=meta.get("chunk_index"),  # type: ignore[arg-type]
                content_snippet=str(meta.get("content", ""))[:300],
            )
        )

    return AskResponse(
        answer=rag_response.answer,
        confidence_score=rag_response.confidence_score,
        latency_ms=rag_response.latency_ms,
        sources=sources,
    )


# ---------------------------------------------------------------------------
# /documents
# ---------------------------------------------------------------------------


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(request: Request) -> DocumentsResponse:
    """List all documents that have been indexed.

    Returns:
        :class:`DocumentsResponse` containing metadata for every indexed file.
    """
    doc_registry: Dict[str, Dict[str, Any]] = request.app.state.doc_registry
    documents = [
        DocumentInfo(
            doc_id=doc_id,
            filename=str(info["filename"]),
            num_chunks=int(info["num_chunks"]),
        )
        for doc_id, info in doc_registry.items()
    ]
    return DocumentsResponse(documents=documents)


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str, request: Request) -> Dict[str, str]:
    """Remove a document and all its chunks from the vector store.

    This uses ChromaDB's metadata filtering to delete only the chunks that
    belong to the specified document, leaving all other documents intact.

    Args:
        doc_id: The document identifier to delete.
        request: FastAPI request (used to access ``app.state``).

    Returns:
        Confirmation dict ``{"status": "deleted", "doc_id": doc_id}``.

    Raises:
        HTTPException 404: If *doc_id* is not found in the registry.
    """
    doc_registry: Dict[str, Dict[str, Any]] = request.app.state.doc_registry

    if doc_id not in doc_registry:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    request.app.state.vectorstore.delete_documents_by_doc_id(doc_id)
    del doc_registry[doc_id]
    logger.info("Deleted document doc_id=%s", doc_id)
    return {"status": "deleted", "doc_id": doc_id}


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return service health status and vector store statistics.

    Returns:
        :class:`HealthResponse` with ``status="ok"`` and the total number of
        stored vectors.
    """
    stats = request.app.state.vectorstore.get_stats()
    doc_count = len(request.app.state.doc_registry)
    return HealthResponse(
        status="ok",
        doc_count=doc_count,
        vector_store_size=stats.get("collection_size", 0),
    )
