"""Vector store package.

Only the abstract base class is imported eagerly.  The ChromaDB implementation
is imported on demand to allow the rest of the package to be imported without
requiring a working ChromaDB installation (e.g. during testing with mocks).
"""

from .base import VectorStore

__all__ = ["VectorStore"]
