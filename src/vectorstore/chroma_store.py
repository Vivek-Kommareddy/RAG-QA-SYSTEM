"""ChromaDB-backed vector store implementation."""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# SQLite compatibility shim
# ChromaDB requires sqlite3 >= 3.35.0.  On macOS / some Linux builds the
# system sqlite3 is too old.  pysqlite3-binary ships its own modern sqlite3
# and can be swapped in transparently before chromadb is imported.
# ---------------------------------------------------------------------------
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass  # pysqlite3-binary not installed – rely on the system sqlite3

import chromadb

from .base import VectorStore
from ..config import get_settings

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Persistent vector store backed by ChromaDB.

    On first use the collection is created; subsequent instantiations re-open
    the existing collection so that data survives restarts.

    Each chunk is stored with a unique UUID identifier.  The ``doc_id`` field
    inside each metadata dict is used to batch-delete chunks belonging to a
    specific source document.
    """

    _COLLECTION_NAME = "rag_collection"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore initialised – collection '%s' contains %d vectors",
            self._COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # VectorStore interface
    # ------------------------------------------------------------------

    def add_documents(
        self,
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Add *embeddings* and their *metadatas* to the ChromaDB collection.

        Args:
            embeddings: Embedding vectors – may be Python lists or numpy arrays.
            metadatas: Metadata dicts (must include ``doc_id``).
        """
        if not embeddings:
            return
        ids = [str(uuid.uuid4()) for _ in embeddings]
        # ChromaDB requires plain Python floats
        processed = [list(map(float, emb)) for emb in embeddings]
        # ChromaDB metadata values must be str/int/float/bool – cast anything else
        clean_meta = [_sanitise_metadata(m) for m in metadatas]
        self._collection.add(ids=ids, embeddings=processed, metadatas=clean_meta)
        logger.debug("Added %d chunks to collection", len(embeddings))

    def search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Return the *top_k* most similar chunks to *query_embedding*.

        ChromaDB returns cosine *distances* (0 = identical, 2 = opposite).
        We convert to similarity as ``1 - distance / 2`` so that the score
        lies in [0, 1].

        Args:
            query_embedding: Query vector.
            top_k: Number of results to return.

        Returns:
            List of ``(metadata, similarity)`` pairs, sorted by descending
            similarity.
        """
        count = self._collection.count()
        if count == 0:
            return []
        k = min(top_k, count)
        embedding_list = list(map(float, query_embedding))
        result = self._collection.query(
            query_embeddings=[embedding_list],
            n_results=k,
            include=["metadatas", "distances"],
        )
        metadatas_list: List[Dict[str, Any]] = result["metadatas"][0]  # type: ignore[index]
        distances: List[float] = result["distances"][0]  # type: ignore[index]
        # Convert cosine distance → similarity score in [0, 1]
        pairs = [(m, 1.0 - d / 2.0) for m, d in zip(metadatas_list, distances)]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def delete_documents_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks whose ``doc_id`` metadata field equals *doc_id*.

        Args:
            doc_id: Document identifier to remove.
        """
        self._collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted chunks for doc_id='%s'", doc_id)

    def delete_collection(self) -> None:
        """Wipe the entire collection and recreate an empty one."""
        self._client.delete_collection(self._COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            self._COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection '%s' cleared", self._COLLECTION_NAME)

    def get_stats(self) -> Dict[str, Any]:
        """Return the number of vectors currently stored.

        Returns:
            Dict with key ``collection_size``.
        """
        return {"collection_size": self._collection.count()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitise_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure all metadata values are ChromaDB-compatible scalar types.

    ChromaDB only accepts ``str``, ``int``, ``float``, and ``bool``.  Any
    other type is converted to its string representation.

    Args:
        meta: Raw metadata dictionary.

    Returns:
        Sanitised metadata dictionary.
    """
    clean: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean
