from __future__ import annotations

from datetime import datetime
import time
from typing import Any

import structlog

from app.core.config import settings
from app.db.repositories.document import search_document_chunks
from app.db.session import AsyncSessionLocal
from app.mcp_server.tools.base import Tool
from app.services.embeddings import EmbeddingService
from app.services.rag_metrics import rag_metrics
from app.services.reranker import RerankerService

logger = structlog.get_logger()


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "MUST be called first for every question without exception. "
        "Searches the user's private knowledge base of uploaded documents."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language search query to retrieve relevant document passages.",
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.embeddings = EmbeddingService()
        self.reranker = RerankerService()

    @staticmethod
    def _coerce_top_k(value: Any) -> int:
        if isinstance(value, int):
            requested = value
        elif isinstance(value, str) and value.strip().isdigit():
            requested = int(value.strip())
        else:
            requested = settings.rag_default_top_k
        requested = max(1, requested)
        return min(requested, settings.rag_max_top_k)

    @staticmethod
    def _coerce_threshold(value: Any) -> float:
        try:
            if value is None:
                return float(settings.rag_similarity_threshold)
            return float(value)
        except (TypeError, ValueError):
            return float(settings.rag_similarity_threshold)

    @staticmethod
    def _format_source(created_at: Any, filename: str, page_number: int | None, section_heading: str | None) -> str:
        if isinstance(created_at, datetime):
            created_str = created_at.date().isoformat()
        else:
            created_str = str(created_at)
        loc_parts: list[str] = []
        if page_number is not None:
            loc_parts.append(f"page {page_number}")
        if isinstance(section_heading, str) and section_heading.strip():
            loc_parts.append(f"section {section_heading.strip()}")
        location = f" ({', '.join(loc_parts)})" if loc_parts else ""
        return f"Source: {filename}{location}, uploaded {created_str}"

    async def run(self, args: dict[str, Any]) -> str:
        """
        IMPORTANT RULE:
        - This method MUST NEVER raise an exception.
        - On any error, it must return a string.
        """
        try:
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return "RAG search not used: missing or invalid query."

            top_k = self._coerce_top_k(args.get("top_k"))
            threshold = self._coerce_threshold(args.get("similarity_threshold"))
            metadata_filter = args.get("metadata_filter")
            if not isinstance(metadata_filter, dict):
                metadata_filter = None

            embed_started = time.perf_counter()
            emb = await self.embeddings.embed_text(query.strip())
            embedding_latency_ms = int((time.perf_counter() - embed_started) * 1000)
            if emb is None:
                rag_metrics.record_embedding_failure()
                return "Retrieval unavailable: embedding error."

            is_hybrid = settings.hybrid_search_enabled
            use_reranker = self.reranker.enabled

            # Over-fetch when reranker is enabled so it has more candidates.
            fetch_k = (
                top_k * settings.reranker_overfetch_multiplier
                if use_reranker
                else top_k
            )

            query_started = time.perf_counter()
            async with AsyncSessionLocal() as session:
                rows = await search_document_chunks(
                    session,
                    query_vector=emb.vector,
                    query_text=query.strip() if is_hybrid else None,
                    top_k=fetch_k,
                    metadata_filter=metadata_filter,
                )
            pgvector_query_ms = int((time.perf_counter() - query_started) * 1000)

            if not rows:
                rag_metrics.record_retrieval(
                    top_similarity=0.0,
                    embedding_latency_ms=embedding_latency_ms,
                    pgvector_query_ms=pgvector_query_ms,
                    returned_chunks=0,
                )
                return "No relevant information found in knowledge base."

            # In hybrid mode the score is an RRF value (different scale from
            # cosine similarity), so the dense threshold is not meaningful.
            effective_threshold = 0.0 if is_hybrid else threshold
            filtered_rows = [(chunk, doc, sim) for chunk, doc, sim in rows if sim > effective_threshold]
            below_threshold_count = len(rows) - len(filtered_rows)

            top_similarity = max(sim for _, _, sim in rows)
            if not filtered_rows:
                rag_metrics.record_retrieval(
                    top_similarity=top_similarity,
                    embedding_latency_ms=embedding_latency_ms,
                    pgvector_query_ms=pgvector_query_ms,
                    returned_chunks=0,
                )
                return "No relevant information found in knowledge base."

            # --- Reranking stage ---
            if use_reranker and len(filtered_rows) > 1:
                candidate_texts = [chunk.text.strip() for chunk, _, _ in filtered_rows]
                rerank_results = await self.reranker.rerank(
                    query=query.strip(),
                    documents=candidate_texts,
                    top_n=top_k,
                )
                # Rebuild filtered_rows in reranked order.
                reranked: list[tuple] = []
                for orig_idx, rerank_score in rerank_results:
                    if orig_idx < len(filtered_rows):
                        chunk, doc, initial_sim = filtered_rows[orig_idx]
                        reranked.append((chunk, doc, rerank_score))
                if reranked:
                    filtered_rows = reranked[:top_k]
                else:
                    filtered_rows = filtered_rows[:top_k]
                score_label = "Rerank-score"
            else:
                filtered_rows = filtered_rows[:top_k]
                score_label = "Similarity"

            blocks: list[str] = []
            for chunk, doc, similarity in filtered_rows:
                source_line = self._format_source(
                    created_at=doc.created_at,
                    filename=doc.filename,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                )
                content = chunk.text.strip().replace("\n\n\n", "\n\n")
                blocks.append(
                    "\n".join(
                        [
                            source_line,
                            f"{score_label}: {similarity:.3f}",
                            f"Content: \"{content}\"",
                        ]
                    )
                )

            logger.info(
                "rag_search_executed",
                query=query[:200],
                top_k=top_k,
                threshold=threshold,
                chunks_returned=len(filtered_rows),
                top_similarity=top_similarity,
                below_threshold_count=below_threshold_count,
                embedding_latency_ms=embedding_latency_ms,
                pgvector_query_ms=pgvector_query_ms,
                has_metadata_filter=bool(metadata_filter),
                reranker_used=use_reranker,
            )
            rag_metrics.record_retrieval(
                top_similarity=top_similarity,
                embedding_latency_ms=embedding_latency_ms,
                pgvector_query_ms=pgvector_query_ms,
                returned_chunks=len(filtered_rows),
            )

            return "\n---\n".join(blocks)
        except Exception:
            logger.error("rag_search_failed_unexpectedly", exc_info=True)
            return "Retrieval failed unexpectedly."
