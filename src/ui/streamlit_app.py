"""Streamlit front-end for the RAG Q&A system.

Features
--------
- Sidebar: document upload, retrieval settings (top_k, rerank toggle, model).
- Main area: chat interface with full session-state message history.
- Source attribution: collapsible expanders showing the exact chunk text.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Q&A System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": "user"|"assistant", "content": ..., "sources": [...]}
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs: List[Dict[str, Any]] = []

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _post(path: str, **kwargs: Any) -> Dict[str, Any]:
    """Make a POST request to the backend and return parsed JSON."""
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(f"{API_URL}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]


def _get(path: str) -> Dict[str, Any]:
    """Make a GET request to the backend and return parsed JSON."""
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(f"{API_URL}{path}")
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]


def upload_files(files: List[Any]) -> Dict[str, Any]:
    """Upload files to the /upload endpoint."""
    multipart = [("files", (f.name, f.read(), f.type)) for f in files]
    return _post("/upload", files=multipart)


def ask_question(question: str, top_k: int, rerank: bool) -> Dict[str, Any]:
    """Send a question to the /ask endpoint."""
    return _post("/ask", json={"question": question, "top_k": top_k, "rerank_enabled": rerank})


def fetch_documents() -> List[Dict[str, Any]]:
    """Fetch the list of indexed documents from /documents."""
    data = _get("/documents")
    return data.get("documents", [])  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sidebar – document upload + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("Retrieval")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=20, value=5)
    rerank_enabled = st.toggle("Enable reranking", value=False)

    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop or browse",
        type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Upload & Index", use_container_width=True) and uploaded_files:
        with st.spinner("Uploading and indexing …"):
            try:
                result = upload_files(uploaded_files)
                st.success(
                    f"Indexed **{len(result['doc_ids'])}** document(s) → "
                    f"**{result['total_chunks']}** chunks"
                )
                st.session_state.uploaded_docs = fetch_documents()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

    # Document registry
    st.subheader("Indexed Documents")
    try:
        docs = fetch_documents()
        if docs:
            for doc in docs:
                st.markdown(
                    f"- **{doc['filename']}** `{doc['num_chunks']} chunks` "
                    f"<span style='font-size:0.75rem;color:grey'>{doc['doc_id'][:8]}…</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No documents indexed yet.")
    except Exception:
        st.caption("Could not reach the API.")

# ---------------------------------------------------------------------------
# Main area – chat interface
# ---------------------------------------------------------------------------
st.title("📄 RAG Q&A System")
st.caption("Ask questions about your uploaded documents.")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"🔍 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], start=1):
                    fname = src.get("filename", "unknown")
                    chunk_idx = src.get("chunk_index", "?")
                    page = src.get("page_number")
                    snippet = src.get("content_snippet", "")
                    loc = f"page {page}" if page else f"chunk {chunk_idx}"
                    st.markdown(f"**{i}. {fname}** – {loc}")
                    if snippet:
                        st.markdown(f"> {snippet}")

# Chat input
if question := st.chat_input("Ask a question …"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question, "sources": []})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking …"):
            try:
                response = ask_question(question, top_k=top_k, rerank=rerank_enabled)
                answer = response.get("answer", "No answer returned.")
                sources = response.get("sources", [])
                confidence = response.get("confidence_score", 0.0)
                latency = response.get("latency_ms", 0.0)

                st.markdown(answer)
                st.caption(f"Confidence: {confidence:.2f} | Latency: {latency:.0f} ms")

                if sources:
                    with st.expander(f"🔍 Sources ({len(sources)})"):
                        for i, src in enumerate(sources, start=1):
                            fname = src.get("filename", "unknown")
                            chunk_idx = src.get("chunk_index", "?")
                            page = src.get("page_number")
                            snippet = src.get("content_snippet", "")
                            loc = f"page {page}" if page else f"chunk {chunk_idx}"
                            st.markdown(f"**{i}. {fname}** – {loc}")
                            if snippet:
                                st.markdown(f"> {snippet}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "latency_ms": latency,
                    }
                )

            except Exception as exc:
                error_msg = f"Error: {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg, "sources": []}
                )

# Clear chat button
if st.session_state.messages:
    if st.button("Clear chat", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
