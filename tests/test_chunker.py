"""Tests for the recursive text chunker."""

from __future__ import annotations

import pytest

from src.ingestion.chunker import chunk_documents, _recursive_split
from src.ingestion.loader import Document


# ---------------------------------------------------------------------------
# _recursive_split unit tests
# ---------------------------------------------------------------------------


def test_short_text_not_split() -> None:
    """Text shorter than chunk_size should be returned as-is."""
    result = _recursive_split("Hello world", ["\n\n", "\n", " ", ""], 200, 20)
    assert result == ["Hello world"]


def test_empty_text_returns_empty() -> None:
    result = _recursive_split("", ["\n\n", "\n", " ", ""], 200, 20)
    assert result == []


def test_split_on_paragraphs() -> None:
    """Paragraph boundaries (\n\n) should be preferred over word breaks."""
    text = "First paragraph content here.\n\nSecond paragraph content here.\n\nThird."
    chunks = _recursive_split(text, ["\n\n", "\n", " ", ""], 50, 0)
    # Each paragraph fits within 50 chars, so they should NOT be merged
    assert len(chunks) >= 2
    assert any("First" in c for c in chunks)
    assert any("Second" in c for c in chunks)


def test_chunk_size_respected() -> None:
    """No chunk should exceed chunk_size (best effort for word boundaries)."""
    text = "word " * 200  # 1000 chars
    chunks = _recursive_split(text, ["\n\n", "\n", ". ", " ", ""], 100, 10)
    for chunk in chunks:
        # Allow small overrun at word boundaries (≤ 10 extra chars)
        assert len(chunk) <= 110, f"Chunk too long: {len(chunk)}"


# ---------------------------------------------------------------------------
# chunk_documents tests
# ---------------------------------------------------------------------------


def test_chunk_produces_multiple_chunks(sample_document: Document) -> None:
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1


def test_chunk_metadata_preserved(sample_document: Document) -> None:
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.metadata["filename"] == "sample.txt"
        assert chunk.metadata["file_type"] == "txt"


def test_chunk_index_sequential(sample_document: Document) -> None:
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=20)
    indices = [c.metadata["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_single_short_document_produces_one_chunk(short_document: Document) -> None:
    chunks = chunk_documents([short_document], chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].content == short_document.content


def test_empty_document_produces_no_chunks(empty_document: Document) -> None:
    chunks = chunk_documents([empty_document], chunk_size=500, chunk_overlap=50)
    assert chunks == []


def test_multiple_documents_each_get_chunks(multi_page_documents) -> None:
    chunks = chunk_documents(multi_page_documents, chunk_size=200, chunk_overlap=20)
    assert len(chunks) > len(multi_page_documents)
    # Chunk indices restart at 0 for each parent document
    seen_filenames = {c.metadata["filename"] for c in chunks}
    assert "doc.pdf" in seen_filenames


def test_invalid_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_documents([], chunk_size=0, chunk_overlap=0)


def test_overlap_gte_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        chunk_documents([], chunk_size=100, chunk_overlap=100)


def test_chunk_content_is_non_empty(sample_document: Document) -> None:
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert chunk.content.strip() != ""
