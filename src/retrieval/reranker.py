"""Cross‑encoder based reranking."""

from __future__ import annotations

from typing import List, Tuple, Dict, Any

import numpy as np



class Reranker:
    """Rerank candidate chunks using a cross‑encoder model.

    The cross‑encoder scores each (question, chunk) pair and returns a new
    ordering of the candidates.  Only the top_k results are kept.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """Initialise the reranker lazily.

        The cross‑encoder model is loaded when the reranker is instantiated.  Importing
        inside the constructor avoids loading heavy models at module import time.
        """
        from sentence_transformers import CrossEncoder  # type: ignore

        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, candidates: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        # Prepare inputs for cross‑encoder: list of pairs (question, text)
        sentences = [candidate[0].get("content", "") for candidate in candidates]
        inputs = [(question, s) for s in sentences]
        scores = self.model.predict(inputs)
        # Combine new scores with existing metadata and similarity; we ignore original similarity
        ranked = list(zip(candidates, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        # Extract metadata and convert cross‑encoder scores to similarity-like values
        reranked = [(meta_sim[0], float(score)) for meta_sim, score in ranked]
        return reranked
