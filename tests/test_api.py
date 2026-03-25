"""Integration tests for the FastAPI endpoints."""

from __future__ import annotations

import io
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(api_client: TestClient) -> None:
    resp = api_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "vector_store_size" in data
    assert "doc_count" in data


# ---------------------------------------------------------------------------
# /documents (empty)
# ---------------------------------------------------------------------------


def test_list_documents_empty(api_client: TestClient) -> None:
    resp = api_client.get("/documents")
    assert resp.status_code == 200
    assert resp.json() == {"documents": []}


# ---------------------------------------------------------------------------
# /upload
# ---------------------------------------------------------------------------


def test_upload_txt_file(api_client: TestClient) -> None:
    content = b"This is a test document.\n\nIt has multiple paragraphs.\n\nThird paragraph here."
    resp = api_client.post(
        "/upload",
        files=[("files", ("test.txt", io.BytesIO(content), "text/plain"))],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["doc_ids"]) == 1
    assert data["total_chunks"] >= 1


def test_upload_md_file(api_client: TestClient) -> None:
    content = b"# Title\n\nFirst section.\n\n## Section 2\n\nContent here."
    resp = api_client.post(
        "/upload",
        files=[("files", ("doc.md", io.BytesIO(content), "text/markdown"))],
    )
    assert resp.status_code == 200
    assert len(resp.json()["doc_ids"]) == 1


def test_upload_multiple_files(api_client: TestClient) -> None:
    files = [
        ("files", ("a.txt", io.BytesIO(b"Document A content " * 30), "text/plain")),
        ("files", ("b.txt", io.BytesIO(b"Document B content " * 30), "text/plain")),
    ]
    resp = api_client.post("/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["doc_ids"]) == 2


def test_upload_unsupported_format_returns_422(api_client: TestClient) -> None:
    resp = api_client.post(
        "/upload",
        files=[("files", ("test.csv", io.BytesIO(b"a,b,c"), "text/csv"))],
    )
    assert resp.status_code == 422


def test_upload_no_files_returns_400(api_client: TestClient) -> None:
    # Sending an empty files list should fail
    resp = api_client.post("/upload", files=[])
    # FastAPI will return 422 (validation) or 400 depending on the handler
    assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# /documents after upload
# ---------------------------------------------------------------------------


def test_list_documents_after_upload(api_client: TestClient) -> None:
    api_client.post(
        "/upload",
        files=[("files", ("listed.txt", io.BytesIO(b"Some content here " * 20), "text/plain"))],
    )
    resp = api_client.get("/documents")
    assert resp.status_code == 200
    docs = resp.json()["documents"]
    assert any(d["filename"] == "listed.txt" for d in docs)


# ---------------------------------------------------------------------------
# /ask
# ---------------------------------------------------------------------------


def test_ask_returns_answer(api_client: TestClient) -> None:
    # First upload a document so there is something to retrieve
    api_client.post(
        "/upload",
        files=[("files", ("kb.txt", io.BytesIO(b"The sky is blue. " * 50), "text/plain"))],
    )
    resp = api_client.post("/ask", json={"question": "What color is the sky?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert isinstance(data["confidence_score"], float)
    assert isinstance(data["latency_ms"], float)
    assert isinstance(data["sources"], list)


def test_ask_empty_question_returns_422(api_client: TestClient) -> None:
    resp = api_client.post("/ask", json={"question": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# DELETE /documents/{doc_id}
# ---------------------------------------------------------------------------


def test_delete_document(api_client: TestClient) -> None:
    upload_resp = api_client.post(
        "/upload",
        files=[("files", ("to_delete.txt", io.BytesIO(b"Delete me " * 30), "text/plain"))],
    )
    doc_id = upload_resp.json()["doc_ids"][0]

    del_resp = api_client.delete(f"/documents/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Document should no longer appear in listing
    list_resp = api_client.get("/documents")
    doc_ids = [d["doc_id"] for d in list_resp.json()["documents"]]
    assert doc_id not in doc_ids


def test_delete_nonexistent_doc_returns_404(api_client: TestClient) -> None:
    resp = api_client.delete("/documents/does-not-exist")
    assert resp.status_code == 404
