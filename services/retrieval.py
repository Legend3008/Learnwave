"""
Retrieval & Reranking Service
Multi-stage retrieval pipeline:
  1. Query understanding & expansion
  2. Hybrid search (semantic + keyword)
  3. Reranking via cross-encoder or score fusion
  4. Context assembly with citations
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from config import (
    TOP_K_RETRIEVAL,
    TOP_K_RERANK,
    SIMILARITY_THRESHOLD,
)
from models.schemas import Chunk, RetrievedChunk, Citation, QueryIntent
from services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    """
    Multi-stage retrieval with hybrid search, reranking, and MMR diversity.
    """

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RERANK,
        doc_ids: Optional[list[str]] = None,
    ) -> list[RetrievedChunk]:
        """
        Full retrieval pipeline: search → rerank → deduplicate → return top-K.

        Args:
            query: User's natural language query.
            top_k: Final number of chunks to return.
            doc_ids: Optional filter to specific documents.

        Returns:
            List of RetrievedChunk with scores and metadata.
        """
        # Stage 1: Query understanding
        sub_queries = self._expand_query(query)

        # Stage 2: Hybrid search (semantic + keyword)
        all_results = self._hybrid_search(sub_queries, doc_ids)

        # Stage 3: Score fusion & deduplication
        fused = self._fuse_results(all_results)

        # Stage 4: Reranking
        reranked = self._rerank(query, fused)

        # Stage 5: Apply MMR for diversity
        diverse = self._apply_mmr(reranked, top_k)

        # Stage 6: Filter by threshold (use original cosine similarity score)
        filtered = [r for r in diverse if r.score >= SIMILARITY_THRESHOLD]

        logger.info(
            "Retrieved %d chunks for query '%s' (pre-filter: %d)",
            len(filtered),
            query[:80],
            len(diverse),
        )
        return filtered[:top_k]

    def build_context(
        self,
        retrieved_chunks: list[RetrievedChunk],
        doc_metadata: dict[str, dict],
    ) -> tuple[str, list[Citation]]:
        """
        Assemble retrieved chunks into a coherent context block with citations.

        Args:
            retrieved_chunks: Chunks from retrieve().
            doc_metadata: Mapping of doc_id → document info (filename, title, etc.).

        Returns:
            (context_text, citations)
        """
        context_parts = []
        citations = []

        for i, rc in enumerate(retrieved_chunks):
            chunk = rc.chunk
            doc_info = doc_metadata.get(chunk.doc_id, {})
            filename = doc_info.get("filename", "Unknown")

            # Build citation label
            if chunk.page_number is not None:
                source = f"[Page {chunk.page_number}, Document: {filename}]"
            elif chunk.start_time is not None:
                minutes = int(chunk.start_time // 60)
                seconds = int(chunk.start_time % 60)
                source = f"[Timestamp: {minutes:02d}:{seconds:02d}, Video: {filename}]"
            else:
                source = f"[Document: {filename}]"

            context_parts.append(f"--- Source {i + 1}: {source} ---\n{chunk.text}")

            citations.append(Citation(
                doc_id=chunk.doc_id,
                filename=filename,
                page_number=chunk.page_number,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                text_snippet=chunk.text[:200],
                score=rc.score,
            ))

        context_text = "\n\n".join(context_parts)
        return context_text, citations

    def classify_intent(self, query: str) -> QueryIntent:
        """Classify query intent for optimized processing."""
        query_lower = query.lower().strip()

        summary_keywords = ["summarize", "summary", "overview", "main points", "key takeaways", "tldr"]
        comparison_keywords = ["compare", "difference", "versus", "vs", "differ", "contrast"]
        definition_keywords = ["what is", "define", "meaning of", "what does", "explain the term"]
        analysis_keywords = ["why", "how does", "analyze", "evaluate", "impact", "implications"]

        if any(kw in query_lower for kw in summary_keywords):
            return QueryIntent.SUMMARY
        elif any(kw in query_lower for kw in comparison_keywords):
            return QueryIntent.COMPARISON
        elif any(kw in query_lower for kw in definition_keywords):
            return QueryIntent.DEFINITION
        elif any(kw in query_lower for kw in analysis_keywords):
            return QueryIntent.ANALYSIS
        return QueryIntent.FACTUAL

    # ── Stage 1: Query Expansion ──────────────────────────────────────────

    def _expand_query(self, query: str) -> list[str]:
        """
        Generate sub-queries for better coverage.
        Simple rule-based expansion (LLM-based HyDE can be added later).
        """
        queries = [query]

        # For complex queries, add a simplified version
        words = query.split()
        if len(words) > 10:
            # Extract key noun phrases (simple heuristic)
            simplified = " ".join(w for w in words if len(w) > 3)
            queries.append(simplified)

        return queries

    # ── Stage 2: Hybrid Search ────────────────────────────────────────────

    def _hybrid_search(
        self,
        queries: list[str],
        doc_ids: Optional[list[str]] = None,
    ) -> list[dict]:
        """Run semantic and keyword search for all queries."""
        all_results = []

        for query in queries:
            # Semantic search
            semantic_results = self.vector_store.search(
                query=query,
                top_k=TOP_K_RETRIEVAL,
                doc_ids=doc_ids,
            )
            for r in semantic_results:
                r["search_type"] = "semantic"
            all_results.extend(semantic_results)

            # Keyword search
            keyword_results = self.vector_store.search_by_text(
                query_text=query,
                top_k=TOP_K_RETRIEVAL // 2,
                doc_ids=doc_ids,
            )
            for r in keyword_results:
                r["search_type"] = "keyword"
            all_results.extend(keyword_results)

        return all_results

    # ── Stage 3: Score Fusion ─────────────────────────────────────────────

    def _fuse_results(self, results: list[dict]) -> list[RetrievedChunk]:
        """
        Deduplicate and fuse scores from multiple search passes.
        Uses Reciprocal Rank Fusion (RRF).
        """
        chunk_scores: dict[str, dict] = {}  # chunk_id → {best_result, rrf_score}
        k = 60  # RRF constant

        # Assign ranks within each search type
        semantic = [r for r in results if r.get("search_type") == "semantic"]
        keyword = [r for r in results if r.get("search_type") == "keyword"]

        for rank, result in enumerate(semantic):
            cid = result["chunk_id"]
            rrf = 1.0 / (k + rank + 1)
            if cid not in chunk_scores:
                chunk_scores[cid] = {"result": result, "rrf": 0.0, "orig_score": result["score"]}
            chunk_scores[cid]["rrf"] += rrf
            # Keep the result with higher score
            if result["score"] > chunk_scores[cid]["result"]["score"]:
                chunk_scores[cid]["result"] = result
            chunk_scores[cid]["orig_score"] = max(chunk_scores[cid]["orig_score"], result["score"])

        for rank, result in enumerate(keyword):
            cid = result["chunk_id"]
            rrf = 1.0 / (k + rank + 1)
            if cid not in chunk_scores:
                chunk_scores[cid] = {"result": result, "rrf": 0.0, "orig_score": result.get("score", 0)}
            chunk_scores[cid]["rrf"] += rrf
            if result.get("score", 0) > chunk_scores[cid]["result"].get("score", 0):
                chunk_scores[cid]["result"] = result
            chunk_scores[cid]["orig_score"] = max(chunk_scores[cid]["orig_score"], result.get("score", 0))

        # Sort by RRF score
        sorted_chunks = sorted(
            chunk_scores.values(), key=lambda x: x["rrf"], reverse=True
        )

        retrieved = []
        for entry in sorted_chunks:
            r = entry["result"]
            meta = r.get("metadata", {})
            chunk = Chunk(
                chunk_id=r["chunk_id"],
                doc_id=meta.get("doc_id", ""),
                text=r["text"],
                page_number=meta.get("page_number"),
                section=meta.get("section"),
                chunk_index=meta.get("chunk_index", 0),
                token_count=meta.get("token_count", 0),
                start_time=meta.get("start_time"),
                end_time=meta.get("end_time"),
            )
            retrieved.append(
                RetrievedChunk(chunk=chunk, score=entry["orig_score"], rerank_score=entry["rrf"])
            )

        return retrieved

    # ── Stage 4: Reranking ────────────────────────────────────────────────

    def _rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """
        Reranking layer.
        Currently uses simple score passthrough.
        Plug in a cross-encoder model (e.g., BGE-Reranker, FlashRank) for production.
        """
        # Attempt to use a cross-encoder if available
        try:
            reranked = self._cross_encoder_rerank(query, chunks)
            if reranked:
                return reranked
        except Exception as e:
            logger.debug("Cross-encoder reranking not available: %s", e)

        # Fallback: score-based ordering (already sorted by RRF)
        return chunks

    def _cross_encoder_rerank(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> Optional[list[RetrievedChunk]]:
        """
        Try reranking with a cross-encoder model.
        Returns None if not available.
        """
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, rc.chunk.text) for rc in chunks]
            scores = model.predict(pairs)

            for i, score in enumerate(scores):
                chunks[i].rerank_score = float(score)

            chunks.sort(key=lambda x: x.rerank_score or 0, reverse=True)
            logger.info("Cross-encoder reranking applied successfully")
            return chunks

        except ImportError:
            return None

    # ── Stage 5: MMR Diversity ────────────────────────────────────────────

    def _apply_mmr(
        self,
        chunks: list[RetrievedChunk],
        top_k: int,
        lambda_param: float = 0.7,
    ) -> list[RetrievedChunk]:
        """
        Maximal Marginal Relevance to reduce redundancy in results.

        Args:
            chunks: Pre-ranked chunks.
            top_k: Number to select.
            lambda_param: Trade-off between relevance (1.0) and diversity (0.0).
        """
        if len(chunks) <= top_k:
            return chunks

        selected: list[RetrievedChunk] = [chunks[0]]
        remaining = list(chunks[1:])

        while len(selected) < top_k and remaining:
            best_idx = 0
            best_mmr = -float("inf")

            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.rerank_score or candidate.score

                # Max similarity to already-selected chunks (simple text overlap)
                max_sim = max(
                    self._text_similarity(candidate.chunk.text, s.chunk.text)
                    for s in selected
                )

                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple Jaccard similarity for MMR diversity computation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
