"""Tests for the retriever module."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

from src.retrieval.retriever import Retriever
from tests.conftest import MockEmbedder, MockVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entries(n: int) -> List[Tuple[Dict[str, Any], float]]:
    """Return *n* fake (metadata, similarity) pairs with decreasing similarity."""
    return [({"content": f"doc_{i}", "doc_id": f"id_{i}"}, 1.0 - i * 0.05) for i in range(n)]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_retriever_returns_top_k(mock_embedder: MockEmbedder, mock_vectorstore: MockVectorStore) -> None:
    """Retriever should return exactly top_k results."""
    entries = _make_entries(10)
    for emb, meta in entries:
        mock_vectorstore._entries.append(([0.1] * MockEmbedder.DIM, meta))

    retriever = Retriever(mock_vectorstore, mock_embedder)
    retriever.top_k = 3
    results = retriever.retrieve("test query")
    assert len(results) == 3


def test_retriever_returns_empty_on_empty_store(mock_embedder: MockEmbedder, mock_vectorstore: MockVectorStore) -> None:
    retriever = Retriever(mock_vectorstore, mock_embedder)
    results = retriever.retrieve("anything")
    assert results == []


def test_retriever_results_are_tuples(mock_embedder: MockEmbedder, mock_vectorstore: MockVectorStore) -> None:
    """Each result should be a (metadata_dict, float) tuple."""
    mock_vectorstore._entries.append(([0.1] * MockEmbedder.DIM, {"content": "hello", "doc_id": "x"}))
    retriever = Retriever(mock_vectorstore, mock_embedder)
    results = retriever.retrieve("hello")
    assert len(results) == 1
    meta, score = results[0]
    assert isinstance(meta, dict)
    assert isinstance(score, float)


def test_mmr_returns_correct_count(mock_embedder: MockEmbedder, mock_vectorstore: MockVectorStore) -> None:
    """MMR selection should still return exactly top_k results from the candidate pool."""
    vec = np.ones(MockEmbedder.DIM, dtype=np.float32)
    vec /= np.linalg.norm(vec)

    for i in range(7):
        meta = {"content": f"chunk_{i}", "doc_id": f"doc_{i}"}
        mock_vectorstore._entries.append((vec.tolist(), meta))

    retriever = Retriever(mock_vectorstore, mock_embedder)
    retriever.top_k = 3
    results = retriever.retrieve("test query")
    # MMR must return exactly top_k items
    assert len(results) == 3
    # All returned items must come from the pool
    all_contents = {f"chunk_{i}" for i in range(7)}
    for meta, score in results:
        assert meta["content"] in all_contents


def test_retriever_top_k_capped_by_store_size(mock_embedder: MockEmbedder, mock_vectorstore: MockVectorStore) -> None:
    """If store has fewer items than top_k, return all available."""
    mock_vectorstore._entries.append(([0.5] * MockEmbedder.DIM, {"content": "only one", "doc_id": "a"}))
    retriever = Retriever(mock_vectorstore, mock_embedder)
    retriever.top_k = 10
    results = retriever.retrieve("query")
    assert len(results) == 1
