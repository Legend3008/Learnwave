"""
Embedding Service
Generates text embeddings via Ollama's local embedding API.
Supports batching and caching for performance.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Optional

import numpy as np
import requests

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_TIMEOUT,
    EMBEDDING_BATCH_SIZE,
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generate embeddings via Ollama's local API with caching and retry."""

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_EMBEDDING_MODEL,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}
        self._api_url = f"{self.base_url}/api/embeddings"

    # ── Public API ────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self._call_ollama(text)
        self._cache[cache_key] = embedding
        return embedding

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.
        Uses caching and processes in batches for efficiency.
        """
        results: list[Optional[list[float]]] = [None] * len(texts)
        to_compute: list[tuple[int, str]] = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                to_compute.append((i, text))

        if to_compute:
            logger.info(
                "Computing %d embeddings (%d cached)",
                len(to_compute),
                len(texts) - len(to_compute),
            )

            # Process in batches
            for batch_start in range(0, len(to_compute), EMBEDDING_BATCH_SIZE):
                batch = to_compute[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
                for idx, text in batch:
                    embedding = self._call_ollama(text)
                    results[idx] = embedding
                    self._cache[self._cache_key(text)] = embedding

                # Progress logging
                done = min(batch_start + EMBEDDING_BATCH_SIZE, len(to_compute))
                logger.info("Embedded %d / %d texts", done, len(to_compute))

        return results  # type: ignore

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a query and return as numpy array (convenience method)."""
        return np.array(self.embed_text(query), dtype=np.float32)

    # ── Health Check ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Check if the Ollama embedding service is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    def _call_ollama(self, text: str, max_retries: int = 3) -> list[float]:
        """Call Ollama embedding API with retry logic."""
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    self._api_url,
                    json={"model": self.model, "prompt": text},
                    timeout=self.timeout,
                )
                response.raise_for_status()

                data = response.json()
                embedding = data.get("embedding")

                if embedding is None:
                    raise ValueError(
                        f"No 'embedding' key in Ollama response: {list(data.keys())}"
                    )

                return embedding

            except requests.exceptions.Timeout:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(
                    "Embedding request timed out (attempt %d/%d)",
                    attempt,
                    max_retries,
                )
            except requests.exceptions.ConnectionError:
                last_error = f"Cannot connect to Ollama at {self.base_url}"
                logger.warning(
                    "Cannot connect to Ollama (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    last_error,
                )
            except requests.exceptions.HTTPError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                logger.warning(
                    "Ollama HTTP error (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    last_error,
                )
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "Embedding error (attempt %d/%d): %s",
                    attempt,
                    max_retries,
                    last_error,
                )

            if attempt < max_retries:
                wait = 2 ** attempt
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to generate embedding after {max_retries} attempts: {last_error}"
        )

    def _cache_key(self, text: str) -> str:
        """Generate a deterministic cache key for a text."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
