"""Similarity search and MMR retrieval."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from ..ingestion.embedder import Embedder
from ..vectorstore.base import VectorStore
from ..config import get_settings


logger = logging.getLogger(__name__)


class Retriever:
    """Retrieve relevant chunks given a question embedding.

    This class wraps a vector store and an embedder.  It supports
    configurable MMR (maximal marginal relevance) retrieval to diversify
    results.  The diversity factor controls the trade‑off between relevance
    and diversity.
    """

    def __init__(self, vectorstore: VectorStore, embedder: Embedder) -> None:
        settings = get_settings()
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.top_k = settings.top_k
        self.diversity = 0.5  # fixed diversity factor for MMR

    def _mmr(self, similarities: List[float], embeddings: np.ndarray, k: int) -> List[int]:
        """Perform MMR selection on the candidate set.

        Args:
            similarities: Precomputed similarity scores between query and each document.
            embeddings: 2‑D array of candidate embeddings.
            k: Number of items to select.

        Returns:
            Indices of the selected documents.
        """
        selected: List[int] = []
        candidate_indices = list(range(len(similarities)))
        lambda_param = self.diversity
        while candidate_indices and len(selected) < k:
            if not selected:
                # select the most similar document
                best_idx = max(candidate_indices, key=lambda i: similarities[i])
                selected.append(best_idx)
                candidate_indices.remove(best_idx)
                continue
            # For remaining candidates compute mmr score
            mmr_scores = {}
            for idx in candidate_indices:
                rel = similarities[idx]
                # compute diversity: max similarity with already selected docs
                diversity_scores = [float(np.dot(embeddings[idx], embeddings[j])) for j in selected]
                diversity_score = max(diversity_scores) if diversity_scores else 0.0
                mmr_scores[idx] = lambda_param * rel - (1 - lambda_param) * diversity_score
            best_idx = max(mmr_scores, key=mmr_scores.get)
            selected.append(best_idx)
            candidate_indices.remove(best_idx)
        return selected

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Return top_k metadata and similarity scores for a query.

        Embeds the query, performs a vector similarity search and optionally
        applies MMR to diversify the results.  Returns a list of metadata
        dictionaries and their corresponding similarity scores.

        Args:
            query: The natural-language question to retrieve context for.
            top_k: Optional per-request override for the number of results to
                return.  Falls back to the value from application config if
                ``None``.
        """
        effective_k = top_k if top_k is not None else self.top_k
        query_embedding = self.embedder.embed_query(query)
        # Search the vector store for a generous candidate set (e.g. top 20)
        candidate_k = max(effective_k * 4, 20)
        results = self.vectorstore.search(query_embedding.tolist(), candidate_k)
        if not results:
            return []
        metadatas, sims = zip(*results)
        # Convert to numpy array of embeddings for MMR diversity calculation; we need to reembed the docs.
        # For simplicity we reembed the doc contents from metadata; if content missing, skip MMR.
        try:
            contents = [md.get("content", "") for md in metadatas]
            doc_embeddings = self.embedder.embed_texts(contents)
        except Exception:
            # Fall back to no MMR
            doc_embeddings = None
        selected_indices: List[int]
        if doc_embeddings is not None:
            selected_indices = self._mmr(list(sims), doc_embeddings, effective_k)
        else:
            selected_indices = list(range(min(effective_k, len(results))))
        final = [(metadatas[i], sims[i]) for i in selected_indices]
        return final
