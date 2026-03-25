"""Full RAG chain combining retrieval, reranking and generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from ..retrieval.retriever import Retriever
from ..retrieval.reranker import Reranker
from ..ingestion.embedder import Embedder
from .llm import BaseLLM
from .prompt_templates import render_prompt


@dataclass
class RAGResponse:
    """Response object returned by the RAG chain."""

    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    latency_ms: float


class RAGChain:
    """Combine retriever, optional reranker and LLM into a single pipeline."""

    def __init__(
        self,
        retriever: Retriever,
        llm: BaseLLM,
        embedder: Embedder,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.embedder = embedder

    def ask(
        self,
        question: str,
        top_k: Optional[int] = None,
        rerank_enabled: Optional[bool] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline for a single question.

        Args:
            question: The natural-language question to answer.
            top_k: Optional per-request override for the number of retrieved
                chunks.  Falls back to the application config value if ``None``.
            rerank_enabled: Optional per-request reranking toggle.  If ``None``,
                reranking is performed whenever a :class:`Reranker` is
                configured at startup.
        """
        start_time = time.perf_counter()
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        if not retrieved:
            return RAGResponse(
                answer="I don't know.",
                sources=[],
                confidence_score=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )
        # Optional reranking — use explicit override if provided, else fall back
        # to whether a reranker was configured at startup.
        should_rerank = rerank_enabled if rerank_enabled is not None else (self.reranker is not None)
        if should_rerank and self.reranker is not None:
            retrieved = self.reranker.rerank(question, retrieved)
        # Compose context from retrieved metadata; assume each metadata contains 'content'
        context_chunks = []
        sims = []
        for meta, sim in retrieved:
            text = meta.get("content")
            if text:
                context_chunks.append(text)
                sims.append(sim)
        context = "\n---\n".join(context_chunks)
        prompt = render_prompt(context=context, question=question)
        answer = self.llm.generate(prompt)
        # Compute confidence as average similarity
        confidence = float(sum(sims) / len(sims)) if sims else 0.0
        latency = (time.perf_counter() - start_time) * 1000
        # Attach sources (metadata) list
        sources = [meta for meta, _ in retrieved]
        return RAGResponse(answer=answer, sources=sources, confidence_score=confidence, latency_ms=latency)
