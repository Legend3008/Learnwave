"""
Tests for Learnwave services.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Chunking Engine Tests ─────────────────────────────────────────────────────

class TestChunkingEngine:
    """Tests for the chunking engine."""

    def setup_method(self):
        from services.chunking import ChunkingEngine
        self.engine = ChunkingEngine(chunk_size=100, chunk_overlap=10, min_chunk_size=10)

    def test_recursive_split_short_text(self):
        """Short text should remain as a single chunk."""
        chunks = self.engine._recursive_split("Hello, this is a short text.")
        assert len(chunks) >= 1
        assert "Hello" in chunks[0]

    def test_recursive_split_long_text(self):
        """Long text should be split into multiple chunks."""
        long_text = "This is a sentence. " * 100
        chunks = self.engine._recursive_split(long_text)
        assert len(chunks) > 1

    def test_sliding_window_split(self):
        """Sliding window should produce overlapping chunks."""
        text = " ".join(f"word{i}" for i in range(200))
        chunks = self.engine._sliding_window_split(text)
        assert len(chunks) > 1

    def test_chunk_pdf_pages(self):
        """Should produce chunks from page data."""
        pages = [
            {"page_number": 1, "text": "This is page one with some text content. " * 20},
            {"page_number": 2, "text": "This is page two with different content. " * 20},
        ]
        chunks = self.engine.chunk_pdf_pages("test-doc-id", pages, strategy="recursive")
        assert len(chunks) > 0
        assert all(c.doc_id == "test-doc-id" for c in chunks)

    def test_chunk_video_segments(self):
        """Should merge short video segments into larger chunks."""
        segments = [
            {"text": f"Segment number {i} of the video.", "start_time": i * 5.0, "end_time": (i + 1) * 5.0}
            for i in range(20)
        ]
        chunks = self.engine.chunk_video_segments("test-video-id", segments, target_duration=30.0)
        assert len(chunks) >= 1
        assert all(c.doc_id == "test-video-id" for c in chunks)
        assert all(c.start_time is not None for c in chunks)

    def test_empty_pages_produce_no_chunks(self):
        """Empty pages should not produce chunks."""
        pages = [{"page_number": 1, "text": ""}, {"page_number": 2, "text": "   "}]
        chunks = self.engine.chunk_pdf_pages("test-doc-id", pages)
        # May have 0 child chunks (parent chunks only if children exist)
        child_chunks = [c for c in chunks if not c.metadata.get("is_parent")]
        assert len(child_chunks) == 0

    def test_token_estimation(self):
        """Token estimation should be roughly 1 token per 4 chars."""
        assert self.engine._estimate_tokens("hello world") == 2  # 11 chars // 4

    def test_detect_section_markdown_heading(self):
        """Should detect markdown headings."""
        section = self.engine._detect_section("# Introduction\nThis is the intro.")
        assert section == "Introduction"

    def test_detect_section_uppercase(self):
        """Should detect ALL CAPS headings."""
        section = self.engine._detect_section("INTRODUCTION\nThis is content.")
        assert section == "INTRODUCTION"


# ── Embedding Service Tests ───────────────────────────────────────────────────

class TestEmbeddingService:
    """Tests for the embedding service (mocked Ollama)."""

    def test_cache_key_deterministic(self):
        from services.embedding_service import EmbeddingService
        svc = EmbeddingService()
        key1 = svc._cache_key("hello world")
        key2 = svc._cache_key("hello world")
        assert key1 == key2

    def test_cache_key_different_for_different_text(self):
        from services.embedding_service import EmbeddingService
        svc = EmbeddingService()
        key1 = svc._cache_key("hello")
        key2 = svc._cache_key("world")
        assert key1 != key2


# ── Conversation Memory Tests ─────────────────────────────────────────────────

class TestConversationMemory:
    """Tests for conversation memory."""

    def setup_method(self):
        from services.conversation import ConversationMemory
        self.memory = ConversationMemory()

    def test_create_session(self):
        session_id = self.memory.create_session(["doc1"])
        assert session_id is not None
        session = self.memory.get_session(session_id)
        assert session is not None
        assert session.session_id == session_id
        assert "doc1" in session.doc_ids

    def test_add_and_retrieve_turns(self):
        session_id = self.memory.create_session()
        self.memory.add_turn(session_id, "user", "What is Python?")
        self.memory.add_turn(session_id, "assistant", "Python is a programming language.")

        history = self.memory.get_recent_history(session_id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_delete_session(self):
        session_id = self.memory.create_session()
        self.memory.add_turn(session_id, "user", "test")
        self.memory.delete_session(session_id)
        assert self.memory.get_session(session_id) is None

    def test_full_context_formatting(self):
        session_id = self.memory.create_session()
        self.memory.add_turn(session_id, "user", "Hello")
        self.memory.add_turn(session_id, "assistant", "Hi there!")

        context = self.memory.get_full_context(session_id)
        assert "User: Hello" in context
        assert "Assistant: Hi there!" in context

    def teardown_method(self):
        """Clean up test sessions."""
        for session in self.memory.list_sessions():
            try:
                self.memory.delete_session(session["session_id"])
            except Exception:
                pass


# ── Retrieval Service Tests ───────────────────────────────────────────────────

class TestRetrievalService:
    """Tests for retrieval query understanding."""

    def test_classify_intent_summary(self):
        from services.retrieval import RetrievalService
        from models.schemas import QueryIntent
        # We need a mock vector store; just test the classify_intent method directly
        svc = RetrievalService.__new__(RetrievalService)
        assert svc.classify_intent("Summarize the document") == QueryIntent.SUMMARY
        assert svc.classify_intent("Give me a summary") == QueryIntent.SUMMARY

    def test_classify_intent_comparison(self):
        from services.retrieval import RetrievalService
        from models.schemas import QueryIntent
        svc = RetrievalService.__new__(RetrievalService)
        assert svc.classify_intent("Compare Python and JavaScript") == QueryIntent.COMPARISON

    def test_classify_intent_definition(self):
        from services.retrieval import RetrievalService
        from models.schemas import QueryIntent
        svc = RetrievalService.__new__(RetrievalService)
        assert svc.classify_intent("What is machine learning?") == QueryIntent.DEFINITION

    def test_classify_intent_factual(self):
        from services.retrieval import RetrievalService
        from models.schemas import QueryIntent
        svc = RetrievalService.__new__(RetrievalService)
        assert svc.classify_intent("When was it published?") == QueryIntent.FACTUAL

    def test_text_similarity(self):
        from services.retrieval import RetrievalService
        svc = RetrievalService.__new__(RetrievalService)
        sim = svc._text_similarity("hello world foo", "hello world bar")
        assert 0 < sim < 1
        assert svc._text_similarity("abc", "abc") == 1.0

    def test_extract_page_range_supports_digits_and_words(self):
        from services.retrieval import RetrievalService
        svc = RetrievalService.__new__(RetrievalService)
        assert svc._extract_page_range("tell me the contents of page 3") == (3, 3)
        assert svc._extract_page_range("What is in the page three of this document?") == (3, 3)
        assert svc._extract_page_range("summarize pages 2 to 4") == (2, 4)

    def test_direct_page_lookup_uses_exact_page_chunks(self):
        from services.retrieval import RetrievalService

        vector_store = MagicMock()
        vector_store.get_chunks.return_value = [{
            "chunk_id": "chunk-1",
            "text": "Page 3 content",
            "metadata": {
                "doc_id": "doc-1",
                "page_number": 3,
                "chunk_index": 0,
                "token_count": 10,
            },
            "score": 1.0,
        }]

        svc = RetrievalService(vector_store)
        results = svc.retrieve("tell me the contents of page three", doc_ids=["doc-1"])

        assert len(results) == 1
        assert results[0].chunk.page_number == 3
        vector_store.get_chunks.assert_called_once_with(
            doc_ids=["doc-1"],
            page_range=(3, 3),
            include_parents=False,
        )
        vector_store.search.assert_not_called()
        vector_store.search_by_text.assert_not_called()

    def test_page_scoped_queries_apply_page_filter_to_hybrid_search(self):
        from services.retrieval import RetrievalService

        vector_store = MagicMock()
        vector_store.search.return_value = []
        vector_store.search_by_text.return_value = []

        svc = RetrievalService(vector_store)
        svc.retrieve("What does page 3 say about algebra?", doc_ids=["doc-1"])

        assert vector_store.search.called
        assert vector_store.search_by_text.called
        assert vector_store.search.call_args.kwargs["page_range"] == (3, 3)
        assert vector_store.search_by_text.call_args.kwargs["page_range"] == (3, 3)


# ── Schema / Model Tests ─────────────────────────────────────────────────────

class TestSchemas:
    """Tests for Pydantic models."""

    def test_document_metadata_defaults(self):
        from models.schemas import DocumentMetadata, DocumentType, ProcessingStatus
        meta = DocumentMetadata(filename="test.pdf", doc_type=DocumentType.PDF)
        assert meta.doc_id is not None
        assert meta.processing_status == ProcessingStatus.PENDING

    def test_chunk_model(self):
        from models.schemas import Chunk
        chunk = Chunk(doc_id="doc1", text="Hello world", chunk_index=0)
        assert chunk.chunk_id is not None
        assert chunk.token_count == 0

    def test_chat_request(self):
        from models.schemas import ChatRequest
        req = ChatRequest(message="What is AI?")
        assert req.message == "What is AI?"
        assert req.session_id is None
        assert req.stream is False


# ── PDF Ingestion Tests (file-based) ─────────────────────────────────────────

class TestPDFIngestion:
    """Tests for PDF text cleaning and table conversion."""

    def setup_method(self):
        from services.pdf_ingestion import PDFIngestionService
        self.svc = PDFIngestionService()

    def test_clean_text_removes_page_numbers(self):
        text = "Hello world\n\n  42  \n\nMore content"
        cleaned = self.svc._clean_text(text)
        assert "42" not in cleaned or "Hello" in cleaned

    def test_clean_text_normalizes_whitespace(self):
        text = "Hello    world   with   spaces"
        cleaned = self.svc._clean_text(text)
        assert "    " not in cleaned

    def test_tables_to_text_markdown(self):
        tables = [[["Header1", "Header2"], ["val1", "val2"]]]
        result = self.svc._tables_to_text(tables)
        assert "Header1" in result
        assert "|" in result
        assert "---" in result

    def test_ingest_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.svc.ingest(Path("/nonexistent/file.pdf"))

    def test_ingest_non_pdf_raises(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a pdf")
            f.flush()
            with pytest.raises(ValueError, match="Not a PDF"):
                self.svc.ingest(Path(f.name))


# ── Config Tests ──────────────────────────────────────────────────────────────

class TestConfig:
    """Tests for configuration loading."""

    def test_config_imports(self):
        import config
        assert config.CHUNK_SIZE > 0
        assert config.OLLAMA_BASE_URL.startswith("http")
        assert config.API_PORT > 0

    def test_directories_exist(self):
        import config
        assert config.UPLOAD_DIR.exists()
        assert config.DATA_DIR.exists()
