"""FastAPI application entry point.

Initialises all shared components (vector store, embedder, retriever, LLM)
once at startup using FastAPI's lifespan context manager, then mounts the
routes defined in :mod:`src.api.routes`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from ..config import get_settings
from .middleware import RequestLoggingMiddleware, global_exception_handler
from .routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s – %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise shared resources on startup and release them on shutdown."""
    logger.info("Starting up RAG Q&A System …")
    settings = get_settings()

    # Heavy imports done lazily inside lifespan so the module can be imported
    # without triggering ChromaDB/OpenAI/etc. initialisation (useful for tests).
    from ..vectorstore.chroma_store import ChromaVectorStore
    from ..ingestion.embedder import Embedder
    from ..retrieval.retriever import Retriever
    from ..retrieval.reranker import Reranker
    from ..generation.llm import get_llm
    from ..generation.chain import RAGChain

    vectorstore = ChromaVectorStore()
    embedder = Embedder()
    retriever = Retriever(vectorstore, embedder)
    reranker = Reranker() if settings.rerank_enabled else None
    llm = get_llm()
    chain = RAGChain(retriever=retriever, llm=llm, embedder=embedder, reranker=reranker)

    # In-memory registry mapping doc_id → {filename, num_chunks}
    # (In production this would be backed by a persistent store.)
    doc_registry: dict[str, dict[str, object]] = {}

    app.state.vectorstore = vectorstore
    app.state.embedder = embedder
    app.state.retriever = retriever
    app.state.reranker = reranker
    app.state.llm = llm
    app.state.chain = chain
    app.state.doc_registry = doc_registry

    logger.info("Startup complete – provider=%s model=%s", settings.llm_provider, settings.llm_model)
    yield
    logger.info("Shutting down RAG Q&A System")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured :class:`FastAPI` instance ready to be served.
    """
    app = FastAPI(
        title="RAG Q&A System",
        description="Production-ready Retrieval-Augmented Generation API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)
    app.add_exception_handler(Exception, global_exception_handler)  # type: ignore[arg-type]
    app.include_router(router)

    return app


app = create_app()
