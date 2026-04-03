"""
Integration tests for the FastAPI application.
Tests API endpoints with mocked services.
Run with: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def client():
    """Create a test client with mocked services."""
    from fastapi.testclient import TestClient

    # We need to mock the services before importing the app
    with patch("app.EmbeddingService") as mock_embed, \
         patch("app.VectorStore") as mock_vs, \
         patch("app.LLMService") as mock_llm:

        mock_embed_instance = MagicMock()
        mock_embed_instance.is_available.return_value = True
        mock_embed.return_value = mock_embed_instance

        mock_vs_instance = MagicMock()
        mock_vs_instance.is_available.return_value = True
        mock_vs_instance.count.return_value = 42
        mock_vs.return_value = mock_vs_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.is_available.return_value = True
        mock_llm_instance.generate_answer.return_value = "mock answer"
        mock_llm_instance.suggest_questions.return_value = []
        mock_llm_instance.reformulate_query.side_effect = lambda q, _h: q
        mock_llm_instance.generate_answer_stream.return_value = iter(["mock answer"])
        mock_llm.return_value = mock_llm_instance

        from app import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    """Tests for /api/health."""

    def test_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "ollama_connected" in data
        assert "chroma_connected" in data


class TestDocumentEndpoints:
    """Tests for /api/documents/* endpoints."""

    def test_list_documents_empty(self, client):
        response = client.get("/api/documents")
        assert response.status_code == 200

    def test_upload_no_file_returns_422(self, client):
        response = client.post("/api/documents/upload")
        assert response.status_code == 422

    def test_upload_unsupported_type(self, client):
        import io
        response = client.post(
            "/api/documents/upload",
            files={"file": ("test.xyz", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert response.status_code == 400

    def test_delete_nonexistent_document(self, client):
        response = client.delete("/api/documents/nonexistent-id")
        assert response.status_code == 404


class TestChatEndpoints:
    """Tests for /api/chat endpoints."""

    def test_chat_empty_message_returns_400(self, client):
        response = client.post("/api/chat", json={"message": ""})
        assert response.status_code == 400

    def test_chat_empty_whitespace_returns_400(self, client):
        response = client.post("/api/chat", json={"message": "   "})
        assert response.status_code == 400

    def test_get_nonexistent_session(self, client):
        response = client.get("/api/chat/nonexistent-session/history")
        assert response.status_code == 404

    def test_delete_session(self, client):
        response = client.delete("/api/chat/some-session-id")
        assert response.status_code == 200

    def test_chat_direct_page_lookup_uses_exact_page_reader(self, client):
        import app

        app.orchestrator.get_page_contents = MagicMock(return_value={
            "status": "ok",
            "doc_id": "doc-1",
            "filename": "0123456789abcdef0123_Books.pdf",
            "title": "Books",
            "page_count": 38,
            "pages": [
                {"page_number": 3, "text": "Exact text from page three.", "source": "pdf"},
            ],
            "text": "[Page 3]\nExact text from page three.",
        })

        response = client.post(
            "/api/chat",
            json={"message": "tell me the content of page 3", "doc_ids": ["doc-1"]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "Page 3 of Books.pdf contains:" in data["answer"]
        assert "Exact text from page three." in data["answer"]
        app.orchestrator.get_page_contents.assert_called_once_with(
            doc_id="doc-1",
            start_page=3,
            end_page=3,
        )

    def test_chat_uses_session_doc_ids_for_page_lookup_when_request_omits_them(self, client):
        import app

        session_id = app.conversation_memory.create_session(["doc-1"])
        app.orchestrator.get_page_contents = MagicMock(return_value={
            "status": "ok",
            "doc_id": "doc-1",
            "filename": "0123456789abcdef0123_Books.pdf",
            "title": "Books",
            "page_count": 38,
            "pages": [
                {"page_number": 3, "text": "Exact text from page three.", "source": "pdf"},
            ],
            "text": "[Page 3]\nExact text from page three.",
        })

        response = client.post(
            "/api/chat",
            json={"message": "tell me the content of page 3", "session_id": session_id},
        )

        assert response.status_code == 200
        app.orchestrator.get_page_contents.assert_called_once_with(
            doc_id="doc-1",
            start_page=3,
            end_page=3,
        )
