# How to Run LearnWave

Step-by-step guide to getting LearnWave running on your local machine.

---

## Prerequisites

| Tool | Purpose | Install |
|------|---------|---------|
| **Python 3.10+** | Runtime | [python.org](https://www.python.org/downloads/) |
| **Ollama** | Local LLM & Embeddings | [ollama.com/download](https://ollama.com/download) |
| **FFmpeg** *(optional)* | Video → MP3 conversion | `brew install ffmpeg` (macOS) |
| **Tesseract** *(optional)* | OCR for scanned PDFs | `brew install tesseract` (macOS) |

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/Legend3008/Learnwave.git
cd Learnwave
```

---

## Step 2 — Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

**Additional packages for URL import (YouTube, Wikipedia, etc.):**

```bash
pip install beautifulsoup4 youtube-transcript-api lxml
```

**Optional packages:**

```bash
# For video/audio transcription
pip install openai-whisper

# For cross-encoder reranking (improves retrieval quality)
pip install sentence-transformers
```

---

## Step 4 — Install & Start Ollama

1. Download and install Ollama from [ollama.com](https://ollama.com/download).

2. Start the Ollama server:

```bash
ollama serve
```

3. Pull the required models (in a new terminal):

```bash
# Embedding model (required)
ollama pull bge-m3

# LLM for answer generation (required)
ollama pull llama3.2:latest
```

4. Verify Ollama is running:

```bash
curl http://localhost:11434/api/tags
```

You should see a JSON response listing the installed models.

---

## Step 5 — Start LearnWave

```bash
python app.py
```

You should see output like:

```
INFO | Initializing Learnwave services...
INFO | ChromaDB initialized at .../data/chroma_db (collection: learnwave_docs, 0 vectors)
INFO | Learnwave ready — 0 documents, 0 vectors
INFO | Uvicorn running on http://0.0.0.0:8000
```

---

## Step 6 — Open in Browser

Go to **http://localhost:8000** in your browser.

- The main app UI will load automatically.
- The landing page is at **http://localhost:8000/landing**.
- The API docs (Swagger) are at **http://localhost:8000/docs**.

---

## Using LearnWave

### Upload a PDF
1. Click **"+ Add Source"** in the left panel.
2. Click **"Upload files"** or drag & drop a PDF.
3. Wait for processing (parsing → chunking → embedding).
4. Start asking questions in the chat!

### Import a URL
1. Click **"+ Add Source"** → **"Website"**.
2. Paste any URL (YouTube, Wikipedia, arXiv, GitHub, or any website).
3. The backend fetches and processes the content automatically.

### Ask Questions
- Type your question in the chat box.
- Get cited answers with page numbers or timestamps.
- Follow-up questions are context-aware.

---

## Environment Variables (Optional)

All config has sensible defaults. Override via environment variables if needed:

```bash
# Use a different LLM model
export OLLAMA_LLM_MODEL="mistral:latest"

# Use a different embedding model
export OLLAMA_EMBEDDING_MODEL="nomic-embed-text"

# Change the server port
export API_PORT=3000

# Adjust chunking
export CHUNK_SIZE=256
export CHUNK_OVERLAP=32

# Adjust retrieval
export TOP_K_RETRIEVAL=30
export SIMILARITY_THRESHOLD=0.25

# Disable OCR
export OCR_ENABLED=false
```

See [config.py](config.py) for all available options.

---

## Project Structure

```
Learnwave/
├── app.py                  # FastAPI server & API routes
├── config.py               # Centralized configuration
├── requirements.txt        # Python dependencies
├── models/
│   └── schemas.py          # Pydantic data models
├── services/
│   ├── chunking.py         # Text chunking engine
│   ├── conversation.py     # Multi-turn conversation memory
│   ├── embedding_service.py# Ollama embedding client
│   ├── llm_service.py      # Ollama LLM client
│   ├── orchestrator.py     # Document processing pipeline
│   ├── pdf_ingestion.py    # PDF & video parsing
│   ├── retrieval.py        # Hybrid search & reranking
│   ├── url_fetcher.py      # Server-side URL content fetcher
│   └── vector_store.py     # ChromaDB vector database
├── static/
│   ├── index.html          # Main app UI
│   └── landing.html        # Landing page
└── tests/
    ├── test_api.py
    └── test_services.py
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Cannot connect to Ollama` | Make sure `ollama serve` is running in a separate terminal |
| `Model not found` | Run `ollama pull bge-m3` and `ollama pull llama3.2:latest` |
| `No text extracted from PDF` | Install Tesseract for scanned PDFs: `brew install tesseract` |
| `Port 8000 already in use` | Kill the old process: `lsof -ti:8000 \| xargs kill -9` |
| `URL import fails` | Install: `pip install beautifulsoup4 youtube-transcript-api lxml` |
| `Video processing fails` | Install FFmpeg: `brew install ffmpeg` and Whisper: `pip install openai-whisper` |

---

## Quick Test (API)

```bash
# Health check
curl http://localhost:8000/api/health

# Import a Wikipedia article
curl -X POST http://localhost:8000/api/documents/url \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://en.wikipedia.org/wiki/Machine_learning"}'

# Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What is machine learning?"}'
```
