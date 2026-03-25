"""Abstract base class for vector stores."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class VectorStore(ABC):
    """Interface for persistent vector stores used in the RAG pipeline.

    Implementations must provide document addition, similarity search,
    targeted deletion by document ID, full collection reset, and stats.
    """

    @abstractmethod
    def add_documents(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Persist *embeddings* together with their associated *metadatas*.

        Args:
            embeddings: List of embedding vectors (one per document chunk).
            metadatas: Corresponding metadata dicts (must include ``doc_id``).
        """

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return the *top_k* nearest neighbours for *query_embedding*.

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            List of ``(metadata, similarity_score)`` pairs sorted by
            descending similarity.
        """

    @abstractmethod
    def delete_documents_by_doc_id(self, doc_id: str) -> None:
        """Remove all chunks that belong to the document with *doc_id*.

        Args:
            doc_id: The document identifier set in chunk metadata at ingest time.
        """

    @abstractmethod
    def delete_collection(self) -> None:
        """Remove *all* entries from the underlying collection."""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the store.

        Returns:
            Dict containing at least ``collection_size`` (number of stored
            vectors).
        """
