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
