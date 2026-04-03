"""
Pydantic models / schemas for Learnwave API and internal data flow.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    PDF = "pdf"
    VIDEO = "video"
    AUDIO = "audio"
    WEB = "web"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class QueryIntent(str, Enum):
    FACTUAL = "factual"
    SUMMARY = "summary"
    COMPARISON = "comparison"
    ANALYSIS = "analysis"
    DEFINITION = "definition"


# ─── Document Models ─────────────────────────────────────────────────────────

class DocumentMetadata(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    doc_type: DocumentType
    title: Optional[str] = None
    author: Optional[str] = None
    page_count: Optional[int] = None
    file_size_bytes: Optional[int] = None
    upload_time: datetime = Field(default_factory=datetime.utcnow)
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    error_message: Optional[str] = None
    source_url: Optional[str] = None      # For web-imported documents
    url_type: Optional[str] = None        # youtube, wikipedia, arxiv, github, website
    source_url: Optional[str] = None
    url_type: Optional[str] = None  # youtube, wikipedia, arxiv, github, website


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    text: str
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int
    token_count: int = 0
    start_time: Optional[float] = None  # For video/audio chunks
    end_time: Optional[float] = None    # For video/audio chunks
    metadata: dict = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float
    rerank_score: Optional[float] = None


# ─── API Request / Response Models ───────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    doc_ids: Optional[list[str]] = None  # Filter to specific documents
    stream: bool = False

class Citation(BaseModel):
    doc_id: str
    filename: str
    page_number: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text_snippet: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    session_id: str
    suggested_questions: list[str] = Field(default_factory=list)

class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: ProcessingStatus
    message: str

class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: ProcessingStatus
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None

class DocumentSummaryResponse(BaseModel):
    doc_id: str
    filename: str
    summary: str
    key_concepts: list[str] = Field(default_factory=list)
    suggested_questions: list[str] = Field(default_factory=list)

class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata]

class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    citations: list[Citation] = Field(default_factory=list)

class ConversationHistory(BaseModel):
    session_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    doc_ids: list[str] = Field(default_factory=list)
    title: str = "Untitled notebook"

class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    chroma_connected: bool
    documents_count: int
    version: str = "1.0.0"


# ─── URL Import Models ───────────────────────────────────────────────────────

class URLImportRequest(BaseModel):
    url: str
    title: Optional[str] = ""
    content: Optional[str] = ""
    word_count: int = 0
    url_type: str = "website"  # youtube, wikipedia, arxiv, github, website

class URLImportResponse(BaseModel):
    doc_id: str
    title: str
    url_type: str
    status: ProcessingStatus
    chunk_count: int = 0
    word_count: int = 0
    message: str
