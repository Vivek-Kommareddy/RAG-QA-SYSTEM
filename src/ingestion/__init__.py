"""Ingestion package.

Provides document loading, chunking, and embedding utilities.
Heavy dependencies (openai, sentence-transformers) are imported lazily
inside the classes that need them.
"""

from .loader import load_file, Document
from .chunker import chunk_documents

__all__ = [
    "load_file",
    "Document",
    "chunk_documents",
]
