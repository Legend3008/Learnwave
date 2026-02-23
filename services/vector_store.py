"""
Vector Store Service
ChromaDB-backed vector storage with namespace isolation, metadata filtering,
and hybrid search support.
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from models.schemas import Chunk
from services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB vector store with:
    - Per-document namespace isolation via metadata filtering
    - Full metadata storage for rich retrieval
    - Hybrid search capabilities
    """

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialized at %s (collection: %s, %d vectors)",
            CHROMA_DIR,
            CHROMA_COLLECTION_NAME,
            self._collection.count(),
        )

    # ── Indexing ──────────────────────────────────────────────────────────

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """
        Embed and store chunks in the vector database.

        Returns:
            Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        logger.info("Embedding and storing %d chunks...", len(chunks))

        # Prepare data
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [self._chunk_to_metadata(c) for c in chunks]

        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(texts)

        # Upsert into ChromaDB (handles both add and update)
        # ChromaDB accepts batches; process in groups to avoid memory issues
        batch_size = 100
        stored = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
            stored += end - i
            logger.debug("Stored batch %d-%d", i, end)

        logger.info("Successfully stored %d chunks", stored)
        return stored

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document."""
        # Get all chunks for this document
        results = self._collection.get(
            where={"doc_id": doc_id},
            include=[],
        )
        if results["ids"]:
            self._collection.delete(ids=results["ids"])
            count = len(results["ids"])
            logger.info("Deleted %d chunks for document %s", count, doc_id)
            return count
        return 0

    # ── Search ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        doc_ids: Optional[list[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> list[dict]:
        """
        Semantic search over stored chunks.

        Args:
            query: Natural language query.
            top_k: Number of results to return.
            doc_ids: Optional filter to specific document(s).
            page_range: Optional (start_page, end_page) filter.

        Returns:
            List of dicts with keys: chunk_id, text, metadata, score.
        """
        query_embedding = self.embedding_service.embed_text(query)

        # Build metadata filter
        where_filter = self._build_filter(doc_ids, page_range)

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self._collection.count() or 1),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        return self._format_results(results)

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 20,
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Keyword/text-based search (ChromaDB's built-in document search).
        Useful for hybrid search alongside semantic search.
        """
        where_filter = self._build_filter(doc_ids)

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=min(top_k, self._collection.count() or 1),
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error("ChromaDB text query failed: %s", e)
            return []

        return self._format_results(results)

    # ── Stats / Info ──────────────────────────────────────────────────────

    def count(self) -> int:
        """Total number of vectors in the store."""
        return self._collection.count()

    def count_by_document(self, doc_id: str) -> int:
        """Number of chunks for a specific document."""
        results = self._collection.get(where={"doc_id": doc_id}, include=[])
        return len(results["ids"])

    def list_documents(self) -> list[str]:
        """Get unique document IDs in the store."""
        # ChromaDB doesn't have a distinct query; get all and deduplicate
        all_meta = self._collection.get(include=["metadatas"])
        doc_ids = set()
        for meta in all_meta.get("metadatas", []):
            if meta and "doc_id" in meta:
                doc_ids.add(meta["doc_id"])
        return list(doc_ids)

    def is_available(self) -> bool:
        """Check if ChromaDB is operational."""
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────────

    def _chunk_to_metadata(self, chunk: Chunk) -> dict:
        """Convert chunk to flat metadata dict (ChromaDB requires flat types)."""
        meta = {
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }
        if chunk.page_number is not None:
            meta["page_number"] = chunk.page_number
        if chunk.section:
            meta["section"] = chunk.section
        if chunk.start_time is not None:
            meta["start_time"] = chunk.start_time
        if chunk.end_time is not None:
            meta["end_time"] = chunk.end_time

        is_parent = chunk.metadata.get("is_parent", False)
        meta["is_parent"] = is_parent
        meta["strategy"] = chunk.metadata.get("strategy", "unknown")

        return meta

    def _build_filter(
        self,
        doc_ids: Optional[list[str]] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Optional[dict]:
        """Build a ChromaDB where filter."""
        conditions = []

        if doc_ids and len(doc_ids) == 1:
            conditions.append({"doc_id": doc_ids[0]})
        elif doc_ids and len(doc_ids) > 1:
            conditions.append({"doc_id": {"$in": doc_ids}})

        if page_range:
            start, end = page_range
            conditions.append({"page_number": {"$gte": start}})
            conditions.append({"page_number": {"$lte": end}})

        if not conditions:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _format_results(self, results: dict) -> list[dict]:
        """Format ChromaDB results into a clean list of dicts."""
        formatted = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return formatted

        ids = results["ids"][0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, chunk_id in enumerate(ids):
            # ChromaDB returns distances (lower = more similar for cosine)
            # Convert to similarity score
            distance = distances[i] if i < len(distances) else 1.0
            score = 1.0 - distance  # cosine distance → cosine similarity

            formatted.append({
                "chunk_id": chunk_id,
                "text": documents[i] if i < len(documents) else "",
                "metadata": metadatas[i] if i < len(metadatas) else {},
                "score": score,
            })

        return formatted
