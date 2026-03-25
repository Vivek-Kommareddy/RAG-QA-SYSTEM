"""Embedding utilities.

Provides a unified interface for computing text embeddings.  Supports:
- OpenAI ``text-embedding-3-small`` (and any other OpenAI embedding model)
  via the **openai >= 1.0** Python SDK.
- Local ``sentence-transformers`` models (e.g. ``all-MiniLM-L6-v2``).

The provider is inferred from ``settings.embedding_model``: if the model name
matches an OpenAI embedding model name the OpenAI client is used; otherwise
the sentence-transformers library is used.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from ..config import get_settings

logger = logging.getLogger(__name__)

# Known OpenAI embedding model prefixes / exact names
_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-ada-002",
    "text-embedding-3-small",
    "text-embedding-3-large",
}


def _is_openai_model(model_name: str) -> bool:
    """Return ``True`` if *model_name* looks like an OpenAI embedding model."""
    return model_name in _OPENAI_EMBEDDING_MODELS or model_name.startswith("text-embedding-")


class Embedder:
    """Compute vector embeddings for text.

    Supports both the OpenAI API (via ``openai >= 1.0``) and local
    sentence-transformers models.  The model and provider are resolved from
    the application config at instantiation time.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.model_name = settings.embedding_model
        self._use_openai = _is_openai_model(self.model_name)
        self._openai_client = None
        self._local_model = None

        if self._use_openai:
            import openai  # type: ignore

            self._openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            logger.info("Embedder initialised with OpenAI model '%s'", self.model_name)
        else:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._local_model = SentenceTransformer(self.model_name)
            logger.info("Embedder initialised with local model '%s'", self.model_name)

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Call the OpenAI embeddings endpoint (v1+ SDK)."""
        assert self._openai_client is not None
        logger.debug("Requesting OpenAI embeddings for %d texts", len(texts))
        response = self._openai_client.embeddings.create(input=texts, model=self.model_name)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings using a local sentence-transformers model."""
        assert self._local_model is not None
        logger.debug("Computing local embeddings for %d texts", len(texts))
        vectors = self._local_model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.array(vectors, dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Return embeddings for a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            2-D NumPy array of shape ``(len(texts), embedding_dim)``.
        """
        if not texts:
            return np.empty((0,), dtype=np.float32)
        if self._use_openai:
            return self._embed_openai(texts)
        return self._embed_local(texts)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Returns:
            1-D NumPy array of floats.
        """
        return self.embed_texts([query]).squeeze(0)
