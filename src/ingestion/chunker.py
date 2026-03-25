"""Text chunking module.

Implements a ``RecursiveCharacterTextSplitter`` from scratch (no LangChain
dependency).  The splitter tries each separator in priority order and
recursively splits until every piece is within ``chunk_size`` characters.
Adjacent pieces are then merged with overlap so that semantic context is
preserved across chunk boundaries.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from .loader import Document

logger = logging.getLogger(__name__)

# Default separator hierarchy mirrors LangChain's behaviour
_DEFAULT_SEPARATORS: List[str] = ["\n\n", "\n", ". ", " ", ""]


def _split_on_separator(text: str, separator: str) -> List[str]:
    """Split *text* on *separator* and keep non-empty parts."""
    if separator == "":
        return list(text)
    return [s for s in text.split(separator) if s]


def _merge_splits(splits: List[str], separator: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Merge small splits into chunks of at most *chunk_size* characters.

    Adjacent splits are concatenated (with *separator* between them) until the
    combined length would exceed *chunk_size*.  An overlap of *chunk_overlap*
    characters is carried forward into the next chunk.

    Args:
        splits: Pre-split text fragments.
        separator: The separator used between fragments when joining.
        chunk_size: Maximum characters per output chunk.
        chunk_overlap: Characters to overlap between consecutive chunks.

    Returns:
        List of merged text chunks.
    """
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    sep_len = len(separator)

    for split in splits:
        split_len = len(split)
        # If adding this split exceeds the limit, flush current buffer
        extra = sep_len if current else 0
        if current_len + extra + split_len > chunk_size and current:
            chunk = separator.join(current)
            if chunk.strip():
                chunks.append(chunk.strip())
            # Build overlap: remove pieces from the front until within overlap budget
            while current and current_len > chunk_overlap:
                removed = current.pop(0)
                current_len -= len(removed) + (sep_len if current else 0)
            current_len = sum(len(s) for s in current) + sep_len * max(0, len(current) - 1)

        current.append(split)
        current_len += split_len + (sep_len if len(current) > 1 else 0)

    # Flush remaining pieces
    if current:
        chunk = separator.join(current)
        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursively split *text* until all chunks fit within *chunk_size*.

    Args:
        text: The input text to split.
        separators: Ordered list of separators to try.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        List of text chunks each no longer than *chunk_size* (best effort).
    """
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find first separator that actually splits this text
    separator = separators[-1]  # fall-back: character split
    next_separators: List[str] = []
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            next_separators = separators[i + 1 :]
            break

    splits = _split_on_separator(text, separator)
    merged = _merge_splits(splits, separator, chunk_size, chunk_overlap)

    # Recursively handle any merged chunk that is still too large
    final: List[str] = []
    for piece in merged:
        if len(piece) > chunk_size and next_separators:
            final.extend(_recursive_split(piece, next_separators, chunk_size, chunk_overlap))
        else:
            if piece.strip():
                final.append(piece)

    return final


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: Optional[List[str]] = None,
) -> List[Document]:
    """Split a list of documents into smaller overlapping chunks.

    Uses a recursive character text splitter that respects paragraph and
    sentence boundaries.  Each output ``Document`` inherits its parent's
    metadata and gains a ``chunk_index`` field.

    Args:
        documents: Input documents to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters of overlap between adjacent chunks.
        separators: Custom separator list (uses default hierarchy if ``None``).

    Returns:
        List of chunked ``Document`` objects.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    seps = separators if separators is not None else _DEFAULT_SEPARATORS
    result: List[Document] = []

    for doc in documents:
        raw_chunks = _recursive_split(doc.content, seps, chunk_size, chunk_overlap)
        logger.debug("Document '%s' split into %d chunks", doc.metadata.get("filename"), len(raw_chunks))
        for idx, chunk_text in enumerate(raw_chunks):
            meta = dict(doc.metadata)
            meta["chunk_index"] = idx
            result.append(Document(content=chunk_text, metadata=meta))

    return result
