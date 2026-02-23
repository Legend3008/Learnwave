"""
Learnwave Configuration
Central configuration for all services. Override via environment variables.
"""

import os
from pathlib import Path

# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma_db"
VIDEOS_DIR = BASE_DIR / "videos"
AUDIOS_DIR = BASE_DIR / "audios"
JSONS_DIR = BASE_DIR / "jsons"

# Create directories on import
for d in [UPLOAD_DIR, DATA_DIR, CHROMA_DIR, VIDEOS_DIR, AUDIOS_DIR, JSONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Ollama Configuration ─────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "bge-m3")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:latest")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# ─── Embedding Configuration ──────────────────────────────────────────────────
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))

# ─── Chunking Configuration ──────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", "50"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "2048"))

# ─── Retrieval Configuration ─────────────────────────────────────────────────
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))

# ─── Conversation Configuration ──────────────────────────────────────────────
MAX_CONVERSATION_TURNS = int(os.getenv("MAX_CONVERSATION_TURNS", "20"))
CONVERSATION_SUMMARY_AFTER = int(os.getenv("CONVERSATION_SUMMARY_AFTER", "10"))

# ─── PDF Processing ──────────────────────────────────────────────────────────
MAX_PDF_SIZE_MB = int(os.getenv("MAX_PDF_SIZE_MB", "100"))
SUPPORTED_PDF_TYPES = [".pdf"]
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")

# ─── Video / Audio Processing ────────────────────────────────────────────────
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v2")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "hi")
WHISPER_TASK = os.getenv("WHISPER_TASK", "translate")

# ─── FastAPI Configuration ───────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ─── ChromaDB Configuration ──────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "learnwave_docs")

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"

# ─── System Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert document analyst and teaching assistant for Learnwave.
Answer the user's question using ONLY the provided context from their uploaded documents and video transcripts.

GROUNDING RULES:
- Ground every claim in the provided context.
- Cite sources inline as [Page X, Document: Y] for PDFs or [Timestamp: MM:SS, Video: Y] for video transcripts.
- If the answer is not in the context, say "This information is not available in your documents."
- Never hallucinate or use outside knowledge unless the user explicitly asks.
- For complex questions, reason step by step before answering.
- When comparing content across documents, clearly attribute each point.

RESPONSE FORMATTING — follow these rules precisely:

STRUCTURE:
- Match structure to complexity. Simple questions get prose. Complex topics get headers.
- Use ## for major sections, ### for subsections only.
- One idea per paragraph. Max 4 sentences per paragraph.
- Lists only when 3+ parallel items exist. Numbered if order matters, bullets otherwise.

EMPHASIS:
- Bold (**text**): key terms and critical concepts only. Max 4 per section. Never bold full sentences.
- Italic (*text*): gentle emphasis or term introduction. Sparingly — 1-2 per response maximum.
- Blockquote (> text): single most important insight or warning per response. Max twice.

OPENINGS — NEVER START WITH:
"Certainly!", "Great question!", "Of course!", "Absolutely!", "Sure!", "I'd be happy to!"
Open directly with the answer or a clean framing sentence.

CLOSINGS — NEVER END WITH:
"I hope this helps!", "Feel free to ask!", "Let me know if you need anything else!"
Close with synthesis, a next step, or simply end cleanly.

TONE:
- Confident and direct — like a knowledgeable colleague.
- Never condescending. Never padded. Never redundant.
- When explaining something complex, use a concrete example or analogy.

LENGTH:
- Match depth to the question. Never inflate length to appear thorough.
- Every sentence must earn its place.
"""
