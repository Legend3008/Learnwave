# Learnwave — AI Teaching Assistant

> Upload PDFs and videos, ask natural language questions, and get **accurate, cited, context-aware answers** powered by a local RAG pipeline.

---

## Features

- **PDF Document Intelligence** — Upload PDFs, auto-parse (including scanned/OCR), and query them in natural language
- **Video Transcript Q&A** — Convert lecture videos to searchable transcripts via Whisper
- **Cited Answers** — Every response includes `[Page X, Document: Y]` or `[Timestamp: MM:SS]` inline citations
- **Multi-Turn Conversations** — Session-scoped memory with context-aware follow-up handling
- **Multi-Document Q&A** — Query across all uploaded documents simultaneously
- **Document Comparison** — Compare two documents on a specific topic
- **Auto-Summaries** — AI-generated executive summaries, key concepts, and suggested questions
- **Hybrid Search** — Semantic (dense) + keyword (sparse) retrieval with Reciprocal Rank Fusion
- **Reranking & MMR** — Cross-encoder reranking and Maximal Marginal Relevance for diversity
- **Streaming Responses** — Server-Sent Events (SSE) for real-time token streaming
- **100% Local** — Runs entirely on your machine with Ollama. No data leaves your system.

---

## Architecture

```
PDF/Video Upload
     │
     ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  PDF Ingestion   │────▶│  Chunking Engine  │────▶│ Embedding Service│
│  (PyMuPDF + OCR) │     │  (Recursive/      │     │  (Ollama bge-m3) │
└─────────────────┘     │   Semantic/Sliding)│     └────────┬─────────┘
                        └──────────────────┘              │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   LLM Service    │◀───│  Retrieval +      │◀───│   Vector Store    │
│ (Ollama llama3.2)│     │  Reranking        │     │   (ChromaDB)     │
└────────┬────────┘     └──────────────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  FastAPI Server  │◀───│  Conversation     │
│  (REST + SSE)    │     │  Memory Manager   │
└─────────────────┘     └──────────────────┘
```

---

## Prerequisites

| Dependency | Purpose | Installation |
|---|---|---|
| **Python 3.10+** | Runtime | [python.org](https://www.python.org/) |
| **Ollama** | Local LLM + Embeddings | [ollama.com](https://ollama.com/) |
| **FFmpeg** | Video → MP3 conversion | `brew install ffmpeg` |
| **Tesseract** (optional) | OCR for scanned PDFs | `brew install tesseract` |

### Ollama Models

```bash
# Required: embedding model
ollama pull bge-m3

# Required: LLM for answer generation
ollama pull llama3.2:latest
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/learnwave.git
cd learnwave

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config (optional — defaults work out of the box)
cp .env.example .env

# 5. Start Ollama (in a separate terminal)
ollama serve

# 6. Start the server
python app.py
```

The API is now live at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

---

## API Reference

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/documents/upload` | Upload a PDF or video file |
| `GET` | `/api/documents` | List all documents |
| `GET` | `/api/documents/{id}/status` | Check processing status |
| `GET` | `/api/documents/{id}/summary` | Get AI-generated summary |
| `DELETE` | `/api/documents/{id}` | Remove document + vectors |
| `POST` | `/api/documents/compare` | Compare two documents |

### Chat

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Send question, get cited answer |
| `POST` | `/api/chat/stream` | Stream answer via SSE |
| `GET` | `/api/chat/{session_id}/history` | Get conversation history |
| `GET` | `/api/chat/sessions` | List all sessions |
| `DELETE` | `/api/chat/{session_id}` | Delete a session |

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |

### Example: Upload a PDF

```bash
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@my-document.pdf"
```

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the main topics covered in the document?"}'
```

---

## Project Structure

```
Learnwave/
├── app.py                      # FastAPI application (entry point)
├── config.py                   # Centralized configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore                  # Git ignore rules
│
├── models/
│   ├── __init__.py
│   └── schemas.py              # Pydantic models & API schemas
│
├── services/
│   ├── __init__.py
│   ├── pdf_ingestion.py        # PDF parsing, OCR, table extraction
│   ├── chunking.py             # Multi-strategy chunking engine
│   ├── embedding_service.py    # Ollama embedding with cache & retry
│   ├── vector_store.py         # ChromaDB vector storage
│   ├── retrieval.py            # Hybrid search, reranking, MMR
│   ├── llm_service.py          # LLM prompt engineering & generation
│   ├── conversation.py         # Session memory management
│   └── orchestrator.py         # Pipeline orchestration
│
├── tests/
│   ├── __init__.py
│   ├── test_services.py        # Unit tests for services
│   └── test_api.py             # API integration tests
│
├── # Legacy scripts (original video pipeline)
├── Video_to_mp3.py
├── mp3_to_json.py
├── preprocess_json.py
└── process_incoming.py
```

---

## Configuration

All settings are configurable via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_EMBEDDING_MODEL` | `bge-m3` | Embedding model |
| `OLLAMA_LLM_MODEL` | `llama3.2:latest` | LLM model |
| `CHUNK_SIZE` | `512` | Target chunk size in tokens |
| `TOP_K_RERANK` | `5` | Final chunks returned per query |
| `MAX_PDF_SIZE_MB` | `100` | Max upload file size |
| `API_PORT` | `8000` | Server port |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=services --cov=models
```

---

## Legacy Video Pipeline

The original video-based pipeline is preserved and still works:

1. Place videos in `videos/` directory
2. `python Video_to_mp3.py` — Convert videos to MP3
3. `python mp3_to_json.py` — Transcribe with Whisper
4. `python preprocess_json.py` — Generate embeddings
5. `python process_incoming.py` — Interactive Q&A

> **Note:** The new API (`app.py`) handles PDF + video ingestion through a unified, production-ready interface with a proper REST API.

---

## License

[GNU General Public License v3.0](LICENSE.txt)
