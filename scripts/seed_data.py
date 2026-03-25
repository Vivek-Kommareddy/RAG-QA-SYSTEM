"""Seed the vector store with the sample documents.

Run this script once before using the system to pre-populate ChromaDB with
the three sample documents in ``data/sample_docs/``.

Usage::

    python scripts/seed_data.py
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import uuid

# Ensure the project root is on the path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import get_settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import Embedder
from src.ingestion.loader import load_file
from src.vectorstore.chroma_store import ChromaVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s – %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load all sample documents and index them into ChromaDB."""
    settings = get_settings()
    vectorstore = ChromaVectorStore()
    embedder = Embedder()

    doc_dir = os.path.join(os.path.dirname(__file__), "..", "data", "sample_docs")
    files = sorted(glob.glob(os.path.join(doc_dir, "*")))

    if not files:
        logger.warning("No files found in %s", doc_dir)
        return

    logger.info("Found %d files to ingest", len(files))

    for path in files:
        try:
            docs = load_file(path)
        except ValueError as exc:
            logger.warning("Skipping '%s': %s", os.path.basename(path), exc)
            continue

        chunks = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)
        doc_id = str(uuid.uuid4())

        for chunk in chunks:
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["content"] = chunk.content

        texts = [c.content for c in chunks]
        embeddings = embedder.embed_texts(texts)
        metadatas = [c.metadata for c in chunks]
        vectorstore.add_documents(embeddings.tolist(), metadatas)

        logger.info(
            "Ingested '%s' → doc_id=%s (%d chunks)",
            os.path.basename(path),
            doc_id,
            len(chunks),
        )

    stats = vectorstore.get_stats()
    logger.info("Done. Vector store now contains %d vectors.", stats["collection_size"])


if __name__ == "__main__":
    main()
