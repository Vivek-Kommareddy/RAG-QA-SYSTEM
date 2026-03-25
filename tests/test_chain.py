"""Tests for the full RAG chain."""

from __future__ import annotations

import pytest

from src.generation.chain import RAGChain, RAGResponse
from src.generation.llm import BaseLLM
from src.retrieval.retriever import Retriever
from tests.conftest import MockEmbedder, MockVectorStore


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _FixedRetriever(Retriever):
    """Returns a fixed set of results without touching a vector store."""

    def __init__(self, results):
        # Bypass parent __init__ – we provide results directly
        self._results = results

    def retrieve(self, query: str, top_k=None):
        return self._results


class _EchoLLM(BaseLLM):
    """Returns a fixed answer regardless of the prompt."""

    def generate(self, prompt: str) -> str:
        return "test answer"


class _EmptyRetriever(Retriever):
    """Always returns an empty result set."""

    def __init__(self):
        pass

    def retrieve(self, query: str, top_k=None):
        return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chain_returns_rag_response(mock_embedder: MockEmbedder) -> None:
    retriever = _FixedRetriever([({"content": "some context"}, 0.9)])
    chain = RAGChain(retriever=retriever, llm=_EchoLLM(), embedder=mock_embedder)
    response = chain.ask("What is this about?")
    assert isinstance(response, RAGResponse)


def test_chain_answer_matches_llm_output(mock_embedder: MockEmbedder) -> None:
    retriever = _FixedRetriever([({"content": "context here"}, 0.8)])
    chain = RAGChain(retriever=retriever, llm=_EchoLLM(), embedder=mock_embedder)
    response = chain.ask("question")
    assert response.answer == "test answer"


def test_chain_populates_sources(mock_embedder: MockEmbedder) -> None:
    retriever = _FixedRetriever([
        ({"content": "chunk 1", "doc_id": "abc"}, 0.9),
        ({"content": "chunk 2", "doc_id": "abc"}, 0.7),
    ])
    chain = RAGChain(retriever=retriever, llm=_EchoLLM(), embedder=mock_embedder)
    response = chain.ask("question")
    assert len(response.sources) == 2


def test_chain_confidence_is_average_similarity(mock_embedder: MockEmbedder) -> None:
    retriever = _FixedRetriever([
        ({"content": "a"}, 0.8),
        ({"content": "b"}, 0.6),
    ])
    chain = RAGChain(retriever=retriever, llm=_EchoLLM(), embedder=mock_embedder)
    response = chain.ask("q")
    assert abs(response.confidence_score - 0.7) < 1e-6


def test_chain_latency_is_positive(mock_embedder: MockEmbedder) -> None:
    retriever = _FixedRetriever([({"content": "x"}, 0.5)])
    chain = RAGChain(retriever=retriever, llm=_EchoLLM(), embedder=mock_embedder)
    response = chain.ask("q")
    assert response.latency_ms >= 0


def test_chain_empty_retrieval_returns_dont_know(mock_embedder: MockEmbedder) -> None:
    """When no chunks are retrieved the chain should admit it doesn't know."""
    chain = RAGChain(retriever=_EmptyRetriever(), llm=_EchoLLM(), embedder=mock_embedder)
    response = chain.ask("unknowable question")
    assert "don't know" in response.answer.lower() or response.answer == ""
    assert response.confidence_score == 0.0
    assert response.sources == []


def test_chain_with_reranker(mock_embedder: MockEmbedder) -> None:
    """Chain should still function when a reranker is provided."""
    from unittest.mock import MagicMock

    retriever = _FixedRetriever([({"content": "ctx"}, 0.7)])
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [({"content": "ctx"}, 0.9)]

    chain = RAGChain(retriever=retriever, llm=_EchoLLM(), embedder=mock_embedder, reranker=mock_reranker)
    response = chain.ask("question with rerank")
    assert response.answer == "test answer"
    mock_reranker.rerank.assert_called_once()
