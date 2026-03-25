"""Shared pytest fixtures for the RAG Q&A test suite."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.ingestion.loader import Document
from src.ingestion.embedder import Embedder
from src.vectorstore.base import VectorStore  # base class only – no chromadb import


# ---------------------------------------------------------------------------
# Sample document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_document() -> Document:
    """A medium-length document suitable for chunking tests."""
    content = (
        "Artificial intelligence is transforming many industries.\n\n"
        "Machine learning models can now process natural language with remarkable accuracy.\n\n"
        "Retrieval-augmented generation combines search and generation into a single pipeline.\n\n"
        "Vector databases store embeddings for efficient similarity search.\n\n"
        "The transformer architecture underlies most modern language models.\n\n"
    ) * 5  # ~1 600 chars
    return Document(content=content, metadata={"filename": "sample.txt", "file_type": "txt"})


@pytest.fixture
def short_document() -> Document:
    """A document shorter than a typical chunk size."""
    return Document(
        content="This is a very short document.",
        metadata={"filename": "short.txt", "file_type": "txt"},
    )


@pytest.fixture
def empty_document() -> Document:
    """An empty document (edge case)."""
    return Document(content="", metadata={"filename": "empty.txt", "file_type": "txt"})


@pytest.fixture
def multi_page_documents() -> List[Document]:
    """Multiple documents simulating PDF pages."""
    return [
        Document(
            content=f"Page {i} content. " * 50,
            metadata={"filename": "doc.pdf", "file_type": "pdf", "page_number": i},
        )
        for i in range(1, 4)
    ]


# ---------------------------------------------------------------------------
# Mock embedder and vector store
# ---------------------------------------------------------------------------


class MockEmbedder(Embedder):
    """An :class:`Embedder` that returns deterministic fake vectors."""

    DIM = 8  # small dimensionality for tests

    def __init__(self) -> None:
        # Skip Embedder.__init__ to avoid loading real models
        self.model_name = "mock"
        self._use_openai = False
        self._openai_client = None
        self._local_model = None

    def embed_texts(self, texts: List[str]) -> np.ndarray:  # type: ignore[override]
        """Return a unique-ish deterministic vector for each text."""
        vectors = []
        for text in texts:
            seed = sum(ord(c) for c in text[:20]) % 1000
            rng = np.random.default_rng(seed)
            v = rng.random(self.DIM).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            vectors.append(v)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:  # type: ignore[override]
        return self.embed_texts([query]).squeeze(0)


class MockVectorStore(VectorStore):
    """In-memory vector store for tests."""

    def __init__(self) -> None:
        self._entries: List[Tuple[List[float], Dict[str, Any]]] = []

    def add_documents(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        for emb, meta in zip(embeddings, metadatas):
            self._entries.append((emb, meta))

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        q = np.array(query_embedding, dtype=np.float32)
        scored = []
        for emb, meta in self._entries:
            e = np.array(emb, dtype=np.float32)
            dot = float(np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e) + 1e-9))
            scored.append((meta, dot))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete_documents_by_doc_id(self, doc_id: str) -> None:
        self._entries = [(e, m) for e, m in self._entries if m.get("doc_id") != doc_id]

    def delete_collection(self) -> None:
        self._entries.clear()

    def get_stats(self) -> Dict[str, Any]:
        return {"collection_size": len(self._entries)}


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    return MockEmbedder()


@pytest.fixture
def mock_vectorstore() -> MockVectorStore:
    return MockVectorStore()


# ---------------------------------------------------------------------------
# FastAPI TestClient fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def api_client(mock_embedder: MockEmbedder, mock_vectorstore: MockVectorStore):
    """Return a FastAPI TestClient with fully mocked components (no ChromaDB/LLM needed)."""
    from contextlib import asynccontextmanager
    from typing import AsyncGenerator

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.retrieval.retriever import Retriever
    from src.generation.chain import RAGChain
    from src.generation.llm import BaseLLM
    from src.api.middleware import RequestLoggingMiddleware, global_exception_handler
    from src.api.routes import router

    class _MockLLM(BaseLLM):
        def generate(self, prompt: str) -> str:
            return "mock answer"

    @asynccontextmanager
    async def _test_lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
        """Lifespan that injects mocks directly – no real ChromaDB or LLM."""
        application.state.embedder = mock_embedder
        application.state.vectorstore = mock_vectorstore
        application.state.retriever = Retriever(mock_vectorstore, mock_embedder)
        application.state.llm = _MockLLM()
        application.state.chain = RAGChain(
            retriever=application.state.retriever,
            llm=application.state.llm,
            embedder=mock_embedder,
            reranker=None,
        )
        application.state.doc_registry = {}
        yield

    test_app = FastAPI(lifespan=_test_lifespan)
    test_app.add_middleware(RequestLoggingMiddleware)
    test_app.add_exception_handler(Exception, global_exception_handler)  # type: ignore[arg-type]
    test_app.include_router(router)

    with TestClient(test_app) as client:
        yield client
