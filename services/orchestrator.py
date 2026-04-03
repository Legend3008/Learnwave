"""
Document Processing Orchestrator
Coordinates the full pipeline: ingestion → chunking → embedding → storage.
Manages document metadata and provides a unified interface for the API layer.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from config import DATA_DIR, UPLOAD_DIR
from models.schemas import (
    DocumentMetadata,
    DocumentType,
    ProcessingStatus,
)
from services.pdf_ingestion import PDFIngestionService, VideoIngestionService
from services.chunking import ChunkingEngine
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService

logger = logging.getLogger(__name__)

METADATA_DIR = DATA_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)


class DocumentOrchestrator:
    """
    Orchestrates the full document processing pipeline and manages document state.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        llm_service: LLMService,
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.pdf_ingestion = PDFIngestionService()
        self.video_ingestion = VideoIngestionService()
        self.chunking_engine = ChunkingEngine()
        self._metadata_cache: dict[str, DocumentMetadata] = {}
        self._load_all_metadata()

    # ── Public API ────────────────────────────────────────────────────────

    def process_pdf(
        self,
        file_path: Path,
        chunking_strategy: str = "recursive",
    ) -> DocumentMetadata:
        """
        Full pipeline: PDF → parse → chunk → embed → store.

        Args:
            file_path: Path to the uploaded PDF.
            chunking_strategy: "recursive", "sliding_window", or "structure_aware".

        Returns:
            DocumentMetadata with processing status.
        """
        file_path = Path(file_path)
        logger.info("Processing PDF: %s", file_path.name)

        try:
            # Step 1: Parse PDF
            metadata, pages = self.pdf_ingestion.ingest(file_path)
            self._save_metadata(metadata)

            if metadata.processing_status == ProcessingStatus.FAILED:
                return metadata

            # Step 2: Chunk
            chunks = self.chunking_engine.chunk_pdf_pages(
                doc_id=metadata.doc_id,
                pages=pages,
                strategy=chunking_strategy,
            )

            if not chunks:
                metadata.processing_status = ProcessingStatus.FAILED
                metadata.error_message = "No chunks produced from document"
                self._save_metadata(metadata)
                return metadata

            # Step 3: Embed and store
            stored_count = self.vector_store.add_chunks(chunks)

            # Step 4: Mark complete
            metadata.processing_status = ProcessingStatus.COMPLETED
            self._save_metadata(metadata)

            logger.info(
                "PDF processed: %s → %d pages → %d chunks → %d stored",
                file_path.name,
                len(pages),
                len(chunks),
                stored_count,
            )
            return metadata

        except Exception as e:
            logger.error("Failed to process PDF %s: %s", file_path.name, e)
            if "metadata" in locals():
                metadata.processing_status = ProcessingStatus.FAILED
                metadata.error_message = str(e)[:500]
                self._save_metadata(metadata)
                return metadata
            raise

    def process_video(
        self,
        video_path: Path,
        language: str = "hi",
        task: str = "translate",
    ) -> DocumentMetadata:
        """
        Full pipeline: Video → audio → transcribe → chunk → embed → store.
        """
        from config import AUDIOS_DIR

        video_path = Path(video_path)
        logger.info("Processing video: %s", video_path.name)

        try:
            # Step 1: Convert to audio
            audio_path = self.video_ingestion.convert_video_to_audio(
                video_path, AUDIOS_DIR
            )

            # Step 2: Transcribe
            metadata, segments = self.video_ingestion.transcribe_audio(
                audio_path, language=language, task=task
            )
            self._save_metadata(metadata)

            # Step 3: Chunk transcript segments
            chunks = self.chunking_engine.chunk_video_segments(
                doc_id=metadata.doc_id,
                segments=segments,
            )

            if not chunks:
                metadata.processing_status = ProcessingStatus.FAILED
                metadata.error_message = "No chunks produced from transcript"
                self._save_metadata(metadata)
                return metadata

            # Step 4: Embed and store
            stored_count = self.vector_store.add_chunks(chunks)

            metadata.processing_status = ProcessingStatus.COMPLETED
            self._save_metadata(metadata)

            logger.info(
                "Video processed: %s → %d segments → %d chunks → %d stored",
                video_path.name,
                len(segments),
                len(chunks),
                stored_count,
            )
            return metadata

        except Exception as e:
            logger.error("Failed to process video %s: %s", video_path.name, e)
            if "metadata" in locals():
                metadata.processing_status = ProcessingStatus.FAILED
                metadata.error_message = str(e)[:500]
                self._save_metadata(metadata)
                return metadata
            raise

    def process_web_content(
        self,
        title: str,
        content: str,
        url: str,
        url_type: str = "website",
    ) -> tuple[DocumentMetadata, int]:
        """
        Full pipeline: Web text → chunk → embed → store.

        Args:
            title: Page title or article name.
            content: Pre-fetched text content from the URL.
            url: Source URL for reference.
            url_type: Type of URL (youtube, wikipedia, arxiv, github, website).

        Returns:
            Tuple of (DocumentMetadata, chunk_count).
        """
        logger.info("Processing web content: %s [%s]", title, url_type)

        try:
            # Build metadata
            metadata = DocumentMetadata(
                filename=title,
                doc_type=DocumentType.WEB,
                title=title,
                source_url=url,
                url_type=url_type,
                file_size_bytes=len(content.encode("utf-8")),
                processing_status=ProcessingStatus.PROCESSING,
            )
            self._save_metadata(metadata)

            # Chunk the text
            chunks = self.chunking_engine.chunk_text(
                doc_id=metadata.doc_id,
                text=content,
                strategy="recursive",
                source_label=url_type,
            )

            if not chunks:
                metadata.processing_status = ProcessingStatus.FAILED
                metadata.error_message = "No chunks produced from web content"
                self._save_metadata(metadata)
                return metadata, 0

            # Embed and store
            stored_count = self.vector_store.add_chunks(chunks)

            metadata.processing_status = ProcessingStatus.COMPLETED
            self._save_metadata(metadata)

            logger.info(
                "Web content processed: '%s' → %d chunks → %d stored",
                title,
                len(chunks),
                stored_count,
            )
            return metadata, stored_count

        except Exception as e:
            logger.error("Failed to process web content '%s': %s", title, e)
            if "metadata" in locals():
                metadata.processing_status = ProcessingStatus.FAILED
                metadata.error_message = str(e)[:500]
                self._save_metadata(metadata)
                return metadata, 0
            raise

    def generate_summary(self, doc_id: str) -> dict:
        """Generate summary + concepts + suggested questions for a document."""
        metadata = self.get_metadata(doc_id)
        if not metadata:
            raise ValueError(f"Document {doc_id} not found")

        # Get all chunks for this document
        results = self.vector_store.search(
            query="main topics and key points",
            top_k=10,
            doc_ids=[doc_id],
        )

        full_text = "\n\n".join(r["text"] for r in results)

        summary = self.llm_service.generate_summary(full_text, metadata.filename)
        concepts = self.llm_service.extract_concepts(full_text)
        questions = self.llm_service.suggest_questions(full_text)

        return {
            "doc_id": doc_id,
            "filename": metadata.filename,
            "summary": summary,
            "key_concepts": concepts,
            "suggested_questions": questions,
        }

    def get_page_contents(
        self,
        doc_id: str,
        start_page: int,
        end_page: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Read exact page content from a stored PDF.

        This is a deterministic fallback for page-specific questions like
        "what is on page 3?" where vector similarity is unnecessary.
        """
        metadata = self.get_metadata(doc_id)
        if not metadata or metadata.doc_type != DocumentType.PDF:
            return None

        end_page = end_page or start_page
        if start_page > end_page:
            start_page, end_page = end_page, start_page

        page_count = metadata.page_count or 0
        if start_page < 1 or (page_count and end_page > page_count):
            return {
                "status": "out_of_range",
                "doc_id": doc_id,
                "filename": metadata.filename,
                "title": metadata.title or metadata.filename,
                "page_count": page_count,
                "pages": [],
                "text": "",
            }

        file_path = self._get_document_file_path(metadata)
        pages: list[dict] = []

        for page_number in range(start_page, end_page + 1):
            text = ""
            source = "none"

            if file_path and file_path.exists():
                text = self._extract_pdf_page_text(file_path, page_number)
                if text.strip():
                    source = "pdf"

            if not text.strip():
                chunks = self.vector_store.get_chunks(
                    doc_ids=[doc_id],
                    page_range=(page_number, page_number),
                    include_parents=False,
                )
                text = "\n\n".join(c["text"] for c in chunks if c.get("text"))
                if text.strip():
                    source = "vector_store"

            cleaned = self._clean_page_text(text)
            pages.append({
                "page_number": page_number,
                "text": cleaned,
                "source": source,
            })

        combined = "\n\n".join(
            f"[Page {page['page_number']}]\n{page['text']}"
            for page in pages
            if page["text"]
        ).strip()

        status = "ok" if combined else "empty"
        return {
            "status": status,
            "doc_id": doc_id,
            "filename": metadata.filename,
            "title": metadata.title or metadata.filename,
            "page_count": page_count,
            "pages": pages,
            "text": combined,
        }

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and all its vectors."""
        deleted = self.vector_store.delete_document(doc_id)
        self._metadata_cache.pop(doc_id, None)

        meta_path = METADATA_DIR / f"{doc_id}.json"
        if meta_path.exists():
            meta_path.unlink()

        logger.info("Deleted document %s (%d chunks)", doc_id, deleted)
        return True

    def get_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """Get document metadata."""
        return self._metadata_cache.get(doc_id)

    def list_documents(self) -> list[DocumentMetadata]:
        """List all processed documents."""
        return list(self._metadata_cache.values())

    def get_document_status(self, doc_id: str) -> dict:
        """Get processing status and chunk count for a document."""
        metadata = self.get_metadata(doc_id)
        if not metadata:
            return {"error": "Document not found"}

        chunk_count = self.vector_store.count_by_document(doc_id)

        return {
            "doc_id": doc_id,
            "filename": metadata.filename,
            "status": metadata.processing_status.value,
            "page_count": metadata.page_count,
            "chunk_count": chunk_count,
            "error_message": metadata.error_message,
        }

    def get_doc_metadata_map(self, doc_ids: Optional[list[str]] = None) -> dict[str, dict]:
        """Get metadata map for citation building."""
        result = {}
        for doc_id, meta in self._metadata_cache.items():
            if doc_ids is None or doc_id in doc_ids:
                result[doc_id] = {
                    "filename": meta.filename,
                    "title": meta.title,
                    "doc_type": meta.doc_type.value,
                }
        return result

    # ── Metadata Persistence ──────────────────────────────────────────────

    def _save_metadata(self, metadata: DocumentMetadata):
        """Save document metadata to disk."""
        self._metadata_cache[metadata.doc_id] = metadata
        meta_path = METADATA_DIR / f"{metadata.doc_id}.json"
        try:
            data = metadata.model_dump(mode="json")
            meta_path.write_text(json.dumps(data, indent=2, default=str))
        except Exception as e:
            logger.error("Failed to save metadata: %s", e)

    def _load_all_metadata(self):
        """Load all document metadata from disk on startup."""
        for path in METADATA_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                meta = DocumentMetadata(**data)
                self._metadata_cache[meta.doc_id] = meta
            except Exception as e:
                logger.warning("Failed to load metadata %s: %s", path.name, e)

        logger.info("Loaded %d document metadata records", len(self._metadata_cache))

    def _get_document_file_path(self, metadata: DocumentMetadata) -> Optional[Path]:
        """Resolve the on-disk file for an uploaded document."""
        if not metadata.filename:
            return None
        candidate = UPLOAD_DIR / metadata.filename
        return candidate if candidate.exists() else None

    def _extract_pdf_page_text(self, file_path: Path, page_number: int) -> str:
        """Extract text from a specific PDF page using pypdf."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))
            if page_number < 1 or page_number > len(reader.pages):
                return ""

            text = reader.pages[page_number - 1].extract_text() or ""
            return text
        except Exception as e:
            logger.warning(
                "Exact page extraction failed for %s page %d: %s",
                file_path.name,
                page_number,
                e,
            )
            return ""

    def _clean_page_text(self, text: str) -> str:
        """Normalize page text for display and prompting."""
        text = text.replace("\x00", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
