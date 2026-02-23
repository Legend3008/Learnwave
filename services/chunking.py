"""
Chunking Engine
Implements multiple chunking strategies: recursive character splitting,
semantic chunking, structure-aware chunking, and sliding window.
Handles both PDF page content and video transcript segments.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from models.schemas import Chunk

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE

logger = logging.getLogger(__name__)


class ChunkingEngine:
    """
    Multi-strategy chunking pipeline.

    Strategies:
      - recursive: Recursive character text splitting (default for PDFs)
      - semantic: Group sentences by semantic similarity
      - sliding_window: Fixed-size windows with overlap
      - structure_aware: Respect document headings and sections
      - video_transcript: Merge short video segments into coherent chunks
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chunk_size: int = MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    # ── Public API ────────────────────────────────────────────────────────

    def chunk_pdf_pages(
        self,
        doc_id: str,
        pages: list[dict],
        strategy: str = "recursive",
    ) -> list[Chunk]:
        """
        Chunk extracted PDF pages into retrieval-sized pieces.

        Args:
            doc_id: Unique document identifier.
            pages: List of page dicts from PDFIngestionService.
            strategy: "recursive", "sliding_window", or "structure_aware".

        Returns:
            List of Chunk objects with full metadata.
        """
        all_chunks: list[Chunk] = []
        chunk_index = 0

        if strategy == "structure_aware":
            # Merge all pages into a single document, preserving page markers
            full_text, page_map = self._merge_pages_with_markers(pages)
            raw_chunks = self._structure_aware_split(full_text)
            for text in raw_chunks:
                page_num = self._resolve_page_number(text, page_map)
                section = self._detect_section(text)
                chunk = Chunk(
                    doc_id=doc_id,
                    text=text.strip(),
                    page_number=page_num,
                    section=section,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(text),
                    metadata={"strategy": "structure_aware"},
                )
                all_chunks.append(chunk)
                chunk_index += 1
        else:
            for page in pages:
                page_number = page.get("page_number")
                text = page.get("text", "")
                if not text.strip():
                    continue

                if strategy == "sliding_window":
                    raw_chunks = self._sliding_window_split(text)
                else:
                    raw_chunks = self._recursive_split(text)

                for chunk_text in raw_chunks:
                    if len(chunk_text.strip()) < self.min_chunk_size:
                        continue
                    section = self._detect_section(chunk_text)
                    chunk = Chunk(
                        doc_id=doc_id,
                        text=chunk_text.strip(),
                        page_number=page_number,
                        section=section,
                        chunk_index=chunk_index,
                        token_count=self._estimate_tokens(chunk_text),
                        metadata={"strategy": strategy},
                    )
                    all_chunks.append(chunk)
                    chunk_index += 1

        # Hierarchical: also create parent chunks (larger context)
        parent_chunks = self._create_parent_chunks(doc_id, all_chunks)
        all_chunks.extend(parent_chunks)

        logger.info(
            "Created %d chunks (%d parents) for doc %s using '%s' strategy",
            len(all_chunks) - len(parent_chunks),
            len(parent_chunks),
            doc_id,
            strategy,
        )
        return all_chunks

    def chunk_video_segments(
        self,
        doc_id: str,
        segments: list[dict],
        target_duration: float = 60.0,
    ) -> list[Chunk]:
        """
        Merge short video transcript segments into coherent chunks.

        Args:
            doc_id: Unique document identifier.
            segments: List of transcript segment dicts with start_time, end_time, text.
            target_duration: Target chunk duration in seconds.

        Returns:
            List of Chunk objects with timestamp metadata.
        """
        if not segments:
            return []

        chunks: list[Chunk] = []
        chunk_index = 0
        current_texts: list[str] = []
        current_start: Optional[float] = None
        current_end: float = 0.0

        for seg in segments:
            text = seg.get("text", "").strip()
            start = seg.get("start_time", 0.0)
            end = seg.get("end_time", 0.0)

            if not text:
                continue

            if current_start is None:
                current_start = start

            current_texts.append(text)
            current_end = end

            # Check if we've reached target duration or natural break
            duration = current_end - (current_start or 0)
            merged = " ".join(current_texts)

            if duration >= target_duration or self._estimate_tokens(merged) >= self.chunk_size:
                chunk = Chunk(
                    doc_id=doc_id,
                    text=merged,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(merged),
                    start_time=current_start,
                    end_time=current_end,
                    metadata={"strategy": "video_merge"},
                )
                chunks.append(chunk)
                chunk_index += 1
                current_texts = []
                current_start = None

        # Flush remaining
        if current_texts:
            merged = " ".join(current_texts)
            if len(merged.strip()) >= self.min_chunk_size:
                chunk = Chunk(
                    doc_id=doc_id,
                    text=merged,
                    chunk_index=chunk_index,
                    token_count=self._estimate_tokens(merged),
                    start_time=current_start,
                    end_time=current_end,
                    metadata={"strategy": "video_merge"},
                )
                chunks.append(chunk)

        logger.info(
            "Created %d video chunks from %d segments for doc %s",
            len(chunks),
            len(segments),
            doc_id,
        )
        return chunks

    def chunk_text(
        self,
        doc_id: str,
        text: str,
        strategy: str = "recursive",
        source_label: str = "web",
    ) -> list[Chunk]:
        """
        Chunk a plain text document (e.g. web content) into retrieval-sized pieces.

        Args:
            doc_id: Unique document identifier.
            text: The full text content.
            strategy: "recursive" or "sliding_window".
            source_label: Label for metadata (e.g. "wikipedia", "youtube").

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        if strategy == "sliding_window":
            raw_chunks = self._sliding_window_split(text)
        else:
            raw_chunks = self._recursive_split(text)

        chunks: list[Chunk] = []
        chunk_index = 0

        for chunk_text in raw_chunks:
            if len(chunk_text.strip()) < self.min_chunk_size:
                continue
            section = self._detect_section(chunk_text)
            chunk = Chunk(
                doc_id=doc_id,
                text=chunk_text.strip(),
                section=section,
                chunk_index=chunk_index,
                token_count=self._estimate_tokens(chunk_text),
                metadata={"strategy": strategy, "source": source_label},
            )
            chunks.append(chunk)
            chunk_index += 1

        # Also create parent chunks for larger context
        parent_chunks = self._create_parent_chunks(doc_id, chunks)
        chunks.extend(parent_chunks)

        logger.info(
            "Created %d text chunks (%d parents) for doc %s using '%s' strategy",
            len(chunks) - len(parent_chunks),
            len(parent_chunks),
            doc_id,
            strategy,
        )
        return chunks

    # ── Splitting Strategies ──────────────────────────────────────────────

    def _recursive_split(self, text: str) -> list[str]:
        """
        Recursively split text on natural boundaries.
        Priority: paragraphs > sentences > words > characters.
        """
        separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        return self._recursive_split_impl(text, separators)

    def _recursive_split_impl(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split using the first applicable separator."""
        if self._estimate_tokens(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Hard split by characters as last resort
            return self._hard_split(text)

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator:
            parts = text.split(separator)
        else:
            return self._hard_split(text)

        chunks = []
        current_chunk = ""

        for part in parts:
            candidate = (current_chunk + separator + part) if current_chunk else part

            if self._estimate_tokens(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # If this single part exceeds chunk_size, recursively split it
                if self._estimate_tokens(part) > self.chunk_size:
                    sub_chunks = self._recursive_split_impl(part, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        if current_chunk:
            chunks.append(current_chunk)

        # Apply overlap
        return self._apply_overlap(chunks)

    def _sliding_window_split(self, text: str) -> list[str]:
        """Fixed-size sliding window with configurable overlap."""
        words = text.split()
        # Approximate: 1 token ≈ 0.75 words
        window_words = int(self.chunk_size * 0.75)
        overlap_words = int(self.chunk_overlap * 0.75)
        step = max(1, window_words - overlap_words)

        chunks = []
        for i in range(0, len(words), step):
            chunk_words = words[i : i + window_words]
            chunk = " ".join(chunk_words)
            if len(chunk.strip()) >= self.min_chunk_size:
                chunks.append(chunk)
            if i + window_words >= len(words):
                break

        return chunks

    def _structure_aware_split(self, text: str) -> list[str]:
        """
        Split text respecting document structure (headings, lists).
        Keeps sections together when possible.
        """
        # Split on markdown-style headings or all-caps lines
        heading_pattern = re.compile(
            r"(?=\n#{1,6}\s)|(?=\n[A-Z][A-Z\s]{5,}\n)|(?=\n\d+\.\s+[A-Z])"
        )
        sections = heading_pattern.split(text)
        sections = [s for s in sections if s.strip()]

        chunks = []
        for section in sections:
            if self._estimate_tokens(section) <= self.chunk_size:
                chunks.append(section)
            else:
                # If section is too large, fall back to recursive split
                sub_chunks = self._recursive_split(section)
                chunks.extend(sub_chunks)

        return chunks

    # ── Hierarchical / Parent Chunks ──────────────────────────────────────

    def _create_parent_chunks(
        self, doc_id: str, child_chunks: list[Chunk], group_size: int = 3
    ) -> list[Chunk]:
        """
        Create parent chunks by merging consecutive child chunks.
        Used for Parent-Document Retrieval: retrieve child, return parent for context.
        """
        parent_chunks = []
        parent_index = len(child_chunks)  # Continue index numbering

        for i in range(0, len(child_chunks), group_size):
            group = child_chunks[i : i + group_size]
            if len(group) < 2:
                continue

            merged_text = "\n\n".join(c.text for c in group)
            if self._estimate_tokens(merged_text) > MAX_CHUNK_SIZE:
                continue  # Skip if parent would be too large

            page_numbers = [c.page_number for c in group if c.page_number]
            child_ids = [c.chunk_id for c in group]

            parent = Chunk(
                doc_id=doc_id,
                text=merged_text,
                page_number=min(page_numbers) if page_numbers else None,
                section=group[0].section,
                chunk_index=parent_index,
                token_count=self._estimate_tokens(merged_text),
                start_time=group[0].start_time,
                end_time=group[-1].end_time,
                metadata={
                    "strategy": "parent",
                    "child_ids": child_ids,
                    "is_parent": True,
                },
            )
            parent_chunks.append(parent)
            parent_index += 1

        return parent_chunks

    # ── Utility ───────────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 chars per token for English."""
        return len(text) // 4

    def _hard_split(self, text: str) -> list[str]:
        """Last-resort character-level splitting."""
        char_limit = self.chunk_size * 4  # 4 chars per token
        chunks = []
        for i in range(0, len(text), char_limit):
            chunk = text[i : i + char_limit]
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlap between consecutive chunks."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped = [chunks[0]]
        overlap_chars = self.chunk_overlap * 4  # Approximate

        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_chars:] if len(chunks[i - 1]) > overlap_chars else ""
            overlapped.append(prev_tail + " " + chunks[i])

        return overlapped

    def _detect_section(self, text: str) -> Optional[str]:
        """Try to detect section heading from first line of chunk."""
        first_line = text.strip().split("\n")[0].strip()
        # Check for markdown heading
        heading_match = re.match(r"^#{1,6}\s+(.+)", first_line)
        if heading_match:
            return heading_match.group(1).strip()
        # Check for ALL CAPS heading
        if first_line.isupper() and len(first_line) < 100:
            return first_line
        # Check for numbered heading
        num_match = re.match(r"^\d+\.?\s+([A-Z].{5,})", first_line)
        if num_match:
            return num_match.group(1).strip()
        return None

    def _merge_pages_with_markers(
        self, pages: list[dict]
    ) -> tuple[str, dict[int, int]]:
        """
        Merge all pages into one string, recording character offset → page number map.
        """
        full_text = ""
        page_map: dict[int, int] = {}  # char_offset → page_number
        for page in pages:
            offset = len(full_text)
            page_map[offset] = page.get("page_number", 0)
            full_text += page.get("text", "") + "\n\n"
        return full_text, page_map

    def _resolve_page_number(self, text: str, page_map: dict[int, int]) -> Optional[int]:
        """Find which page a chunk most likely belongs to (best effort)."""
        # This is approximate; for exact mapping we'd need the offset
        # For now return None — the per-page strategy handles this better
        return None
