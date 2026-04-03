"""
PDF Ingestion Service
Handles PDF parsing, text extraction, OCR fallback, table extraction,
and metadata preservation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised in lightweight test envs
    fitz = None

from config import OCR_ENABLED, OCR_LANGUAGE, UPLOAD_DIR
from models.schemas import DocumentMetadata, DocumentType, ProcessingStatus

logger = logging.getLogger(__name__)


class PDFIngestionService:
    """Robust PDF parser with text extraction, OCR fallback, and table handling."""

    def __init__(self):
        self._ocr_available = self._check_ocr()

    # ── Public API ────────────────────────────────────────────────────────

    def ingest(self, file_path: Path) -> tuple[DocumentMetadata, list[dict]]:
        """
        Parse a PDF and return metadata + list of page-level extracted content.

        Returns:
            (metadata, pages)  where each page dict has:
                - page_number: int
                - text: str
                - tables: list[list[list[str]]]
                - has_images: bool
        """
        file_path = Path(file_path)
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is not installed")
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"Not a PDF file: {file_path}")

        logger.info("Ingesting PDF: %s", file_path.name)

        try:
            doc = fitz.open(str(file_path))
        except Exception as e:
            logger.error("Failed to open PDF %s: %s", file_path.name, e)
            raise

        metadata = self._extract_metadata(doc, file_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            page_data = self._extract_page(page, page_num + 1, file_path.name)
            if page_data["text"].strip():  # Only include pages with content
                pages.append(page_data)

        doc.close()

        if not pages:
            metadata.processing_status = ProcessingStatus.FAILED
            metadata.error_message = "No extractable text found in PDF"
            logger.warning("No text extracted from %s", file_path.name)
        else:
            metadata.processing_status = ProcessingStatus.PROCESSING
            logger.info(
                "Extracted %d pages with content from %s",
                len(pages),
                file_path.name,
            )

        return metadata, pages

    # ── Page Extraction ───────────────────────────────────────────────────

    def _extract_page(self, page: fitz.Page, page_number: int, filename: str) -> dict:
        """Extract text, tables, and image info from a single page."""
        text = self._extract_text(page)
        tables = self._extract_tables(page)
        has_images = len(page.get_images(full=True)) > 0

        # OCR fallback: if very little text and there are images
        if len(text.strip()) < 50 and has_images and OCR_ENABLED and self._ocr_available:
            ocr_text = self._ocr_page(page)
            if len(ocr_text.strip()) > len(text.strip()):
                logger.info("Using OCR text for page %d of %s", page_number, filename)
                text = ocr_text

        # Clean the extracted text
        text = self._clean_text(text)

        # Append table content as structured text
        if tables:
            table_text = self._tables_to_text(tables)
            text = text.rstrip() + "\n\n" + table_text

        return {
            "page_number": page_number,
            "text": text,
            "tables": tables,
            "has_images": has_images,
        }

    def _extract_text(self, page: fitz.Page) -> str:
        """Extract text preserving reading order and structure."""
        # Use "text" mode for structured extraction
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        lines = []

        for block in blocks:
            if block["type"] == 0:  # Text block
                for line in block.get("lines", []):
                    spans_text = ""
                    for span in line.get("spans", []):
                        spans_text += span.get("text", "")
                    if spans_text.strip():
                        lines.append(spans_text)
                lines.append("")  # Paragraph break between blocks

        return "\n".join(lines)

    def _extract_tables(self, page: fitz.Page) -> list[list[list[str]]]:
        """Extract tables from page using PyMuPDF's built-in table finder."""
        tables = []
        try:
            tab_finder = page.find_tables()
            for table in tab_finder.tables:
                extracted = table.extract()
                if extracted and len(extracted) > 1:  # At least header + 1 row
                    # Clean cell contents
                    cleaned = []
                    for row in extracted:
                        cleaned_row = [
                            (cell.strip() if cell else "") for cell in row
                        ]
                        cleaned.append(cleaned_row)
                    tables.append(cleaned)
        except Exception as e:
            logger.debug("Table extraction failed on page: %s", e)
        return tables

    def _tables_to_text(self, tables: list[list[list[str]]]) -> str:
        """Convert extracted tables to readable markdown-style text."""
        parts = []
        for i, table in enumerate(tables):
            if not table:
                continue
            parts.append(f"[Table {i + 1}]")
            for row_idx, row in enumerate(table):
                row_text = " | ".join(cell for cell in row)
                parts.append(row_text)
                if row_idx == 0:
                    # Header separator
                    parts.append(" | ".join("---" for _ in row))
            parts.append("")
        return "\n".join(parts)

    # ── OCR ───────────────────────────────────────────────────────────────

    def _check_ocr(self) -> bool:
        """Check if Tesseract OCR is available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
            return True
        except Exception:
            logger.info("Tesseract OCR is not available; OCR fallback disabled")
            return False

    def _ocr_page(self, page: fitz.Page) -> str:
        """Run OCR on a page by rendering it as an image."""
        try:
            import pytesseract
            from PIL import Image
            import io

            # Render page at 300 DPI for good OCR quality
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_bytes))

            text = pytesseract.image_to_string(image, lang=OCR_LANGUAGE)
            return text
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            return ""

    # ── Metadata ──────────────────────────────────────────────────────────

    def _extract_metadata(self, doc: fitz.Document, file_path: Path) -> DocumentMetadata:
        """Extract document metadata from PDF properties."""
        pdf_metadata = doc.metadata or {}

        return DocumentMetadata(
            filename=file_path.name,
            doc_type=DocumentType.PDF,
            title=pdf_metadata.get("title") or file_path.stem,
            author=pdf_metadata.get("author"),
            page_count=len(doc),
            file_size_bytes=file_path.stat().st_size,
        )

    # ── Text Cleaning ─────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        """Clean extracted text: remove artifacts, normalize whitespace."""
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r"[ \t]+", " ", text)
        # Remove lines that are just page numbers
        text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
        # Remove excessive blank lines (keep max 2)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove common header/footer artifacts
        text = re.sub(r"(?i)^(page\s+\d+|confidential|draft)\s*$", "", text, flags=re.MULTILINE)
        return text.strip()


class VideoIngestionService:
    """
    Refactored video ingestion pipeline.
    Converts video → MP3 → transcribed JSON chunks.
    """

    def __init__(self):
        self._check_ffmpeg()

    def _check_ffmpeg(self):
        """Verify FFmpeg is installed."""
        import shutil
        if not shutil.which("ffmpeg"):
            logger.warning("FFmpeg not found. Video processing will not work.")

    def convert_video_to_audio(self, video_path: Path, output_dir: Path) -> Path:
        """Convert a video file to MP3."""
        import subprocess

        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = output_dir / f"{video_path.stem}.mp3"

        if audio_path.exists():
            logger.info("Audio already exists: %s", audio_path.name)
            return audio_path

        logger.info("Converting %s to MP3...", video_path.name)
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", str(video_path), "-y", str(audio_path)],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg error: {result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"FFmpeg timed out converting {video_path.name}")

        logger.info("Converted to: %s", audio_path.name)
        return audio_path

    def transcribe_audio(
        self,
        audio_path: Path,
        language: str = "hi",
        task: str = "translate",
    ) -> tuple[DocumentMetadata, list[dict]]:
        """Transcribe audio using Whisper and return metadata + timestamped chunks."""
        import whisper
        from config import WHISPER_MODEL

        audio_path = Path(audio_path)
        logger.info("Loading Whisper model: %s", WHISPER_MODEL)
        model = whisper.load_model(WHISPER_MODEL)

        logger.info("Transcribing: %s", audio_path.name)
        result = model.transcribe(
            audio=str(audio_path),
            language=language,
            task=task,
            word_timestamps=False,
        )

        metadata = DocumentMetadata(
            filename=audio_path.stem,
            doc_type=DocumentType.VIDEO,
            title=audio_path.stem,
        )

        pages = []  # We treat each segment as a "page" for uniformity
        for segment in result.get("segments", []):
            pages.append({
                "page_number": None,
                "text": segment["text"].strip(),
                "start_time": segment["start"],
                "end_time": segment["end"],
                "tables": [],
                "has_images": False,
            })

        metadata.processing_status = ProcessingStatus.PROCESSING
        return metadata, pages
