"""Evaluate the RAG system on a set of questions."""

from __future__ import annotations

import json
import os
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.config import get_settings
from src.ingestion.embedder import Embedder
from src.vectorstore.chroma_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.retrieval.reranker import Reranker
from src.generation.llm import get_llm
from src.generation.chain import RAGChain


def main() -> None:
    settings = get_settings()
    embedder = Embedder()
    vectorstore = ChromaVectorStore()
    retriever = Retriever(vectorstore, embedder)
    reranker = Reranker() if settings.rerank_enabled else None
    llm = get_llm()
    chain = RAGChain(retriever=retriever, llm=llm, embedder=embedder, reranker=reranker)
    # Define question/answer pairs for evaluation
    qa_pairs: List[Tuple[str, str]] = [
        ("What is the main topic of the AI research paper?", "It discusses transformer efficiency improvements."),
        ("What benefits does the company handbook mention?", "It outlines policies and benefits for employees including PTO."),
        ("How does the product handle data backups?", "The FAQ explains the backup strategy for the SaaS product."),
        # Additional synthetic questions
        ("What is the document about transformers?", "The research paper focuses on making transformers more efficient."),
        ("Describe the PTO policy in the handbook.", "The company handbook details paid time off policies and procedures."),
        ("What support channels are available for the product?", "The FAQ lists email and chat support channels."),
        ("Summarize the AI paper's conclusion.", "The paper concludes that new techniques significantly reduce compute."),
        ("What holidays are mentioned in the handbook?", "The employee handbook specifies observed holidays."),
        ("How can users reset their password?", "The FAQ describes the password reset process."),
        ("What is covered in the handbook benefits section?", "Benefits such as health insurance and 401k are covered.")
    ]
    results = []
    for q, expected in qa_pairs:
        response = chain.ask(q)
        # Compute relevance: cosine similarity between embeddings of question and answer
        q_emb = embedder.embed_query(q).reshape(1, -1)
        a_emb = embedder.embed_query(response.answer).reshape(1, -1)
        relevance = float(cosine_similarity(q_emb, a_emb)[0, 0])
        # Compute faithfulness: percentage of answer sentences contained in sources
        sentences = [s.strip() for s in response.answer.split(".") if s.strip()]
        context = " ".join(meta.get("content", "") for meta in response.sources)
        faithful_sentences = [s for s in sentences if s.lower() in context.lower()]
        faithfulness = len(faithful_sentences) / len(sentences) if sentences else 0.0
        retrieval_precision = 1.0  # Placeholder since ground truth is unknown
        results.append({
            "question": q,
            "expected": expected,
            "answer": response.answer,
            "relevance": relevance,
            "faithfulness": faithfulness,
            "retrieval_precision": retrieval_precision,
        })
    # Print a summary table
    print("Evaluation Results:\n")
    for row in results:
        print(f"Q: {row['question']}")
        print(f"Pred: {row['answer']}")
        print(f"Rel: {row['relevance']:.2f} | Faith: {row['faithfulness']:.2f} | Prec: {row['retrieval_precision']:.2f}\n")

    # Optionally save results to JSON
    out_path = os.path.join(os.path.dirname(__file__), "evaluation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()