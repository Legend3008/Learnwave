"""
Learnwave FastAPI Application
Production-ready REST API for PDF & video document intelligence.

Endpoints:
  POST   /api/documents/upload         — Upload & ingest PDF or video
  GET    /api/documents                — List all documents
  GET    /api/documents/{id}/status    — Processing status
  GET    /api/documents/{id}/summary   — Auto-generated summary
  DELETE /api/documents/{id}           — Remove document + vectors
  POST   /api/chat                     — Send message, get answer
  POST   /api/chat/stream              — Send message, stream answer (SSE)
  GET    /api/chat/{session_id}/history — Conversation history
  GET    /api/health                   — Health check
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    API_HOST,
    API_PORT,
    CORS_ORIGINS,
    LOG_FORMAT,
    LOG_LEVEL,
    MAX_PDF_SIZE_MB,
    SUPPORTED_PDF_TYPES,
    UPLOAD_DIR,
)
from models.schemas import (
    ChatRequest,
    ChatResponse,
    Citation,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentSummaryResponse,
    DocumentUploadResponse,
    HealthResponse,
    ProcessingStatus,
    QueryIntent,
    URLImportRequest,
    URLImportResponse,
)
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from services.llm_service import LLMService
from services.retrieval import RetrievalService
from services.conversation import ConversationMemory
from services.orchestrator import DocumentOrchestrator

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# ── Global Service Instances ──────────────────────────────────────────────────
embedding_service: EmbeddingService
vector_store: VectorStore
llm_service: LLMService
retrieval_service: RetrievalService
conversation_memory: ConversationMemory
orchestrator: DocumentOrchestrator


def _display_document_name(filename: str) -> str:
    """Strip the upload prefix for user-facing messages."""
    return re.sub(r"^[a-f0-9]{20,}_", "", filename or "")


def _format_page_label(start_page: int, end_page: int) -> str:
    """Create a human-readable page label."""
    return f"page {start_page}" if start_page == end_page else f"pages {start_page}-{end_page}"


def _get_effective_doc_ids(
    request_doc_ids: Optional[list[str]],
    session_id: str,
) -> Optional[list[str]]:
    """Prefer request doc_ids, otherwise fall back to the session's bound docs."""
    if request_doc_ids:
        return request_doc_ids

    session = conversation_memory.get_session(session_id)
    if session and session.doc_ids:
        return list(session.doc_ids)

    return None


def _try_exact_page_response(
    message: str,
    doc_ids: Optional[list[str]],
) -> Optional[dict]:
    """
    Answer page-specific PDF questions by reading the exact page content.

    This bypasses fuzzy retrieval for requests like "what is on page 3?"
    and mirrors how a tool-using assistant would inspect the actual page.
    """
    page_range = retrieval_service.extract_page_range(message)
    if not page_range or not doc_ids:
        return None

    if len(doc_ids) != 1:
        return {
            "answer": (
                f"Page-specific questions work best with a single selected document. "
                f"You currently have {len(doc_ids)} sources selected."
            ),
            "citations": [],
            "suggested_questions": [],
        }

    page_result = orchestrator.get_page_contents(
        doc_id=doc_ids[0],
        start_page=page_range[0],
        end_page=page_range[1],
    )
    if not page_result:
        return None

    filename = page_result["filename"]
    display_name = _display_document_name(filename)
    page_label = _format_page_label(page_range[0], page_range[1])

    if page_result["status"] == "out_of_range":
        page_count = page_result.get("page_count") or 0
        verb = "is" if page_range[0] == page_range[1] else "are"
        noun = "page" if page_count == 1 else "pages"
        return {
            "answer": (
                f"{display_name} has {page_count} {noun}, so {page_label} {verb} out of range."
            ),
            "citations": [],
            "suggested_questions": [],
        }

    pages = [page for page in page_result["pages"] if page.get("text")]
    if not pages:
        target = "that page" if page_range[0] == page_range[1] else "those pages"
        return {
            "answer": (
                f"I found {page_label} in {display_name}, but I couldn't extract readable text from {target}."
            ),
            "citations": [],
            "suggested_questions": [],
        }

    citations = [
        Citation(
            doc_id=page_result["doc_id"],
            filename=filename,
            page_number=page["page_number"],
            text_snippet=page["text"][:200],
            score=1.0,
        )
        for page in pages
    ]

    context = "\n\n".join(
        f"--- Source {i + 1}: [Page {page['page_number']}, Document: {filename}] ---\n{page['text']}"
        for i, page in enumerate(pages)
    )

    if retrieval_service.is_direct_page_lookup(message):
        if len(pages) == 1:
            answer = f"Page {pages[0]['page_number']} of {display_name} contains:\n\n{pages[0]['text']}"
        else:
            answer = "\n\n".join(
                f"Page {page['page_number']} of {display_name} contains:\n\n{page['text']}"
                for page in pages
            )
    else:
        answer = llm_service.generate_answer(
            query=message,
            context=context,
            citations=citations,
            conversation_history=None,
            intent=retrieval_service.classify_intent(message),
        )

    suggested_questions = llm_service.suggest_questions(context, message)
    return {
        "answer": answer,
        "citations": citations,
        "suggested_questions": suggested_questions,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    global embedding_service, vector_store, llm_service
    global retrieval_service, conversation_memory, orchestrator

    logger.info("Initializing Learnwave services...")

    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)
    llm_service = LLMService()
    retrieval_service = RetrievalService(vector_store)
    conversation_memory = ConversationMemory()
    orchestrator = DocumentOrchestrator(embedding_service, vector_store, llm_service)

    logger.info(
        "Learnwave ready — %d documents, %d vectors",
        len(orchestrator.list_documents()),
        vector_store.count(),
    )
    yield
    logger.info("Shutting down Learnwave...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Learnwave",
    description="AI Teaching Assistant — Upload PDFs & videos, ask questions, get cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static Files ──────────────────────────────────────────────────────────────
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["System"])
async def root():
    """Serve the Learnwave UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Learnwave</h1><p>static/index.html not found</p>", status_code=500)


@app.get("/landing", response_class=HTMLResponse, tags=["System"])
async def landing():
    """Serve the Learnwave marketing landing page."""
    landing_path = STATIC_DIR / "landing.html"
    if landing_path.exists():
        return HTMLResponse(content=landing_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Learnwave</h1><p>static/landing.html not found</p>", status_code=500)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check — verifies Ollama and ChromaDB connectivity."""
    return HealthResponse(
        status="healthy",
        ollama_connected=embedding_service.is_available() and llm_service.is_available(),
        chroma_connected=vector_store.is_available(),
        documents_count=len(orchestrator.list_documents()),
    )


# ── Document Upload & Management ─────────────────────────────────────────────

@app.post("/api/documents/upload", response_model=DocumentUploadResponse, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF or video file for processing.
    The file will be parsed, chunked, embedded, and stored for Q&A.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    supported = SUPPORTED_PDF_TYPES + [".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav"]

    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {supported}",
        )

    # Check file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_PDF_SIZE_MB} MB",
        )

    # Save to upload directory
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{file.filename}"
    save_path.write_bytes(content)

    try:
        if suffix == ".pdf":
            metadata = orchestrator.process_pdf(save_path)
        elif suffix in [".mp4", ".mkv", ".avi", ".mov", ".mp3", ".wav"]:
            metadata = orchestrator.process_video(save_path)
        else:
            raise HTTPException(status_code=400, detail=f"Cannot process: {suffix}")

        return DocumentUploadResponse(
            doc_id=metadata.doc_id,
            filename=metadata.filename,
            status=metadata.processing_status,
            message=(
                f"Successfully processed {file.filename}"
                if metadata.processing_status == ProcessingStatus.COMPLETED
                else f"Processing failed: {metadata.error_message}"
            ),
        )

    except Exception as e:
        logger.error("Upload processing failed: %s", e)
        # Clean up
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:200]}")


@app.post("/api/documents/url", response_model=URLImportResponse, tags=["Documents"])
async def import_url(request: URLImportRequest):
    """
    Import a web page, YouTube video, Wikipedia article, arXiv paper, or GitHub repo.
    The backend fetches, parses, chunks, and embeds the content — just send the URL.
    Optionally, pre-fetched content can be sent to skip the fetch step.
    """
    if not request.url.strip():
        raise HTTPException(status_code=400, detail="No URL provided")

    from services.url_fetcher import fetch_url, classify_url

    try:
        url = request.url.strip()

        # Server-side fetch: the backend does all the heavy lifting
        if not request.content or len(request.content.strip()) < 50:
            result = fetch_url(url)
            title = result.title
            content = result.content
            url_type = result.url_type
        else:
            # Use pre-supplied content (fallback / legacy)
            title = request.title.strip() if request.title else url
            content = request.content
            url_type = request.url_type or classify_url(url)

        metadata, chunk_count = orchestrator.process_web_content(
            title=title,
            content=content,
            url=url,
            url_type=url_type,
        )

        return URLImportResponse(
            doc_id=metadata.doc_id,
            title=metadata.title or title,
            url_type=url_type,
            status=metadata.processing_status,
            chunk_count=chunk_count,
            word_count=len(content.split()),
            message=(
                f"Successfully imported '{title}'"
                if metadata.processing_status == ProcessingStatus.COMPLETED
                else f"Import failed: {metadata.error_message}"
            ),
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("URL import failed for %s: %s", request.url, e)
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)[:200]}")


@app.get("/api/documents", response_model=DocumentListResponse, tags=["Documents"])
async def list_documents():
    """List all ingested documents."""
    return DocumentListResponse(documents=orchestrator.list_documents())


@app.get("/api/documents/{doc_id}/status", tags=["Documents"])
async def document_status(doc_id: str):
    """Get the processing status and chunk count for a document."""
    status = orchestrator.get_document_status(doc_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail=status["error"])
    return status


@app.get("/api/documents/{doc_id}/summary", response_model=DocumentSummaryResponse, tags=["Documents"])
async def document_summary(doc_id: str):
    """Generate an AI summary, key concepts, and suggested questions for a document."""
    try:
        result = orchestrator.generate_summary(doc_id)
        return DocumentSummaryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Summary generation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)[:200]}")


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
async def delete_document(doc_id: str):
    """Delete a document and all its vectors from the system."""
    metadata = orchestrator.get_metadata(doc_id)
    if not metadata:
        raise HTTPException(status_code=404, detail="Document not found")

    orchestrator.delete_document(doc_id)
    return {"message": f"Document {doc_id} deleted", "filename": metadata.filename}


# ── Chat (Q&A) ───────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Send a natural language question and receive a cited, context-aware answer.
    Supports multi-turn conversation via session_id.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Session management
    session_id = request.session_id or conversation_memory.create_session(request.doc_ids)
    history = conversation_memory.get_recent_history(session_id)
    effective_doc_ids = _get_effective_doc_ids(request.doc_ids, session_id)

    # Page-scoped questions should use the original wording and exact page lookup.
    query = request.message
    page_range = retrieval_service.extract_page_range(request.message)
    if history and not page_range:
        query = llm_service.reformulate_query(query, history)

    exact_response = _try_exact_page_response(request.message, effective_doc_ids)
    if exact_response:
        conversation_memory.add_turn(session_id, "user", request.message)
        conversation_memory.add_turn(
            session_id,
            "assistant",
            exact_response["answer"],
            exact_response["citations"],
        )
        return ChatResponse(
            answer=exact_response["answer"],
            session_id=session_id,
            citations=exact_response["citations"],
            suggested_questions=exact_response["suggested_questions"],
        )

    # Classify intent
    intent = retrieval_service.classify_intent(query)

    # Retrieve relevant chunks
    retrieved = retrieval_service.retrieve(
        query=query,
        doc_ids=effective_doc_ids,
    )

    if not retrieved:
        answer = "I couldn't find relevant information in your documents to answer this question. Try rephrasing or uploading additional documents."
        conversation_memory.add_turn(session_id, "user", request.message)
        conversation_memory.add_turn(session_id, "assistant", answer)
        return ChatResponse(
            answer=answer,
            session_id=session_id,
            citations=[],
            suggested_questions=[],
        )

    # Build context with citations
    doc_metadata = orchestrator.get_doc_metadata_map(effective_doc_ids)
    context, citations = retrieval_service.build_context(retrieved, doc_metadata)

    # Generate answer
    answer = llm_service.generate_answer(
        query=request.message,
        context=context,
        citations=citations,
        conversation_history=history,
        intent=intent,
    )

    # Generate follow-up questions
    suggested = llm_service.suggest_questions(context, request.message)

    # Store conversation turns
    conversation_memory.add_turn(session_id, "user", request.message)
    conversation_memory.add_turn(session_id, "assistant", answer, citations)

    return ChatResponse(
        answer=answer,
        citations=citations,
        session_id=session_id,
        suggested_questions=suggested,
    )


@app.post("/api/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Stream a response token-by-token via Server-Sent Events (SSE).
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = request.session_id or conversation_memory.create_session(request.doc_ids)
    history = conversation_memory.get_recent_history(session_id)
    effective_doc_ids = _get_effective_doc_ids(request.doc_ids, session_id)

    query = request.message
    page_range = retrieval_service.extract_page_range(request.message)
    if history and not page_range:
        query = llm_service.reformulate_query(query, history)

    exact_response = _try_exact_page_response(request.message, effective_doc_ids)
    intent = retrieval_service.classify_intent(query)

    retrieved = []
    citations = []
    context = ""

    if not exact_response:
        retrieved = retrieval_service.retrieve(query=query, doc_ids=effective_doc_ids)
        if not retrieved:
            exact_response = {
                "answer": (
                    "I couldn't find relevant information in your documents to answer this question. "
                    "Try rephrasing or uploading additional documents."
                ),
                "citations": [],
                "suggested_questions": [],
            }
        else:
            doc_metadata = orchestrator.get_doc_metadata_map(effective_doc_ids)
            context, citations = retrieval_service.build_context(retrieved, doc_metadata)

    def event_generator():
        import json as _json

        if exact_response:
            answer = exact_response["answer"]
            yield f"data: {_json.dumps({'token': answer})}\n\n"
            conversation_memory.add_turn(session_id, "user", request.message)
            conversation_memory.add_turn(
                session_id,
                "assistant",
                answer,
                exact_response["citations"],
            )
            final = {
                "done": True,
                "session_id": session_id,
                "citations": [c.model_dump(mode="json") for c in exact_response["citations"]],
            }
            yield f"data: {_json.dumps(final)}\n\n"
            return

        full_response = []
        for token in llm_service.generate_answer_stream(
            query=request.message,
            context=context,
            citations=citations,
            conversation_history=history,
            intent=intent,
        ):
            full_response.append(token)
            yield f"data: {_json.dumps({'token': token})}\n\n"

        # Final event with metadata
        answer = "".join(full_response)
        conversation_memory.add_turn(session_id, "user", request.message)
        conversation_memory.add_turn(session_id, "assistant", answer, citations)

        final = {
            "done": True,
            "session_id": session_id,
            "citations": [c.model_dump(mode="json") for c in citations],
        }
        yield f"data: {_json.dumps(final)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/chat/{session_id}/history", tags=["Chat"])
async def chat_history(session_id: str):
    """Retrieve conversation history for a session."""
    session = conversation_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session.session_id,
        "title": getattr(session, 'title', 'Untitled notebook'),
        "turns": [
            {
                "role": t.role,
                "content": t.content,
                "timestamp": t.timestamp.isoformat(),
                "citations": [c.model_dump(mode="json") for c in t.citations] if t.citations else [],
            }
            for t in session.turns
        ],
    }


@app.get("/api/chat/sessions", tags=["Chat"])
async def list_sessions():
    """List all conversation sessions."""
    return conversation_memory.list_sessions()


@app.delete("/api/chat/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """Delete a conversation session."""
    conversation_memory.delete_session(session_id)
    return {"message": f"Session {session_id} deleted"}


class RenameSessionRequest(BaseModel):
    title: str

@app.patch("/api/chat/{session_id}", tags=["Chat"])
async def rename_session(session_id: str, request: RenameSessionRequest):
    """Rename a conversation session."""
    ok = conversation_memory.rename_session(session_id, request.title)
    if not ok:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "title": request.title}


# ── Cross-Document Comparison ─────────────────────────────────────────────────

class CompareRequest(BaseModel):
    doc_id_a: str
    doc_id_b: str
    topic: str

@app.post("/api/documents/compare", tags=["Documents"])
async def compare_documents(request: CompareRequest):
    """Compare two documents on a specific topic."""
    meta_a = orchestrator.get_metadata(request.doc_id_a)
    meta_b = orchestrator.get_metadata(request.doc_id_b)

    if not meta_a or not meta_b:
        raise HTTPException(status_code=404, detail="One or both documents not found")

    # Retrieve from each document
    chunks_a = retrieval_service.retrieve(
        query=request.topic, top_k=5, doc_ids=[request.doc_id_a]
    )
    chunks_b = retrieval_service.retrieve(
        query=request.topic, top_k=5, doc_ids=[request.doc_id_b]
    )

    doc_metadata = orchestrator.get_doc_metadata_map(
        [request.doc_id_a, request.doc_id_b]
    )

    context_a, _ = retrieval_service.build_context(chunks_a, doc_metadata)
    context_b, _ = retrieval_service.build_context(chunks_b, doc_metadata)

    prompt = f"""Compare the following two documents on the topic: "{request.topic}"

=== Document A: {meta_a.filename} ===
{context_a}

=== Document B: {meta_b.filename} ===
{context_b}

Provide a structured comparison highlighting similarities and differences.
Use a table format where applicable. Cite specific pages/timestamps."""

    comparison = llm_service._call_generate(prompt)

    return {
        "topic": request.topic,
        "document_a": meta_a.filename,
        "document_b": meta_b.filename,
        "comparison": comparison,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level=LOG_LEVEL.lower(),
    )
