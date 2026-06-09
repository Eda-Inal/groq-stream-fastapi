from __future__ import annotations

from datetime import datetime
import time
from typing import Any

import structlog

from app.core.config import settings
from app.db.repositories.document import count_stale_chunks, search_document_chunks
from app.db.session import AsyncSessionLocal
from app.tool_server.tools.base import Tool, ToolResult
from app.services.embeddings import EmbeddingService
from app.services.rag_metrics import rag_metrics
from app.services.reranker import RerankerService

logger = structlog.get_logger()


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search the user's uploaded documents (private RAG knowledge base) "
        "using semantic similarity. "
        "Returns relevant passages with source filename, page number, and similarity score. "
        "Optional hybrid search and reranking are applied internally."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query derived from the user's question. "
                    "Keep all keywords and domain-specific terms from the user's question. "
                    "Do not summarize or shorten — every keyword is a search signal."
                ),
            },
            "top_k": {
                "anyOf": [{"type": "integer"}, {"type": "string"}],
                "description": (
                    "Number of passages to retrieve (integer 3-10). "
                    "Use higher values (8-10) for broad or complex questions, "
                    "lower values (3-5) for narrow or factual questions."
                ),
                "default": 5,
            },
            "similarity_threshold": {
                "anyOf": [{"type": "number"}, {"type": "string"}],
                "description": (
                    "Minimum semantic similarity score for a passage to be included (0.5-0.9). "
                    "Higher values return only closely matching passages; "
                    "lower values cast a wider net. Default 0.7 works for most cases."
                ),
                "default": 0.7,
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
        return max(3, min(10, requested))

    @staticmethod
    def _coerce_threshold(value: Any) -> float:
        try:
            if value is None:
                return float(settings.rag_similarity_threshold)
            return max(0.5, min(0.9, float(value)))
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

    async def run(self, args: dict[str, Any]) -> ToolResult:
        try:
            query = args.get("query")
            if not isinstance(query, str) or not query.strip():
                return ToolResult(ok=False, content="RAG search not used: missing or invalid query.")

            top_k = self._coerce_top_k(args.get("top_k"))
            threshold = self._coerce_threshold(args.get("similarity_threshold"))
            metadata_filter = args.get("metadata_filter")
            if not isinstance(metadata_filter, dict):
                metadata_filter = None

            embed_started = time.perf_counter()
            emb = await self.embeddings.embed_text(
                "Represent this sentence for searching relevant passages: " + query.strip()
            )
            embedding_latency_ms = int((time.perf_counter() - embed_started) * 1000)
            if emb is None:
                rag_metrics.record_embedding_failure()
                return ToolResult(ok=False, content="Retrieval unavailable: embedding error.")

            is_hybrid = settings.hybrid_search_enabled
            use_reranker = self.reranker.enabled

            # Over-fetch when reranker is enabled so it has more candidates.
            fetch_k = (
                top_k * settings.reranker_overfetch_multiplier
                if use_reranker
                else top_k
            )

            current_model = settings.embedding_model_name
            query_started = time.perf_counter()
            async with AsyncSessionLocal() as session:
                rows = await search_document_chunks(
                    session,
                    query_vector=emb.vector,
                    query_text=query.strip() if is_hybrid else None,
                    top_k=fetch_k,
                    metadata_filter=metadata_filter,
                    embedding_model=current_model,
                )
                user_id = metadata_filter.get("user_id") if isinstance(metadata_filter, dict) else None
                stale_count = await count_stale_chunks(
                    session,
                    user_id=user_id,
                    current_model=current_model,
                )
            pgvector_query_ms = int((time.perf_counter() - query_started) * 1000)

            if not rows:
                rag_metrics.record_retrieval(
                    top_similarity=0.0,
                    embedding_latency_ms=embedding_latency_ms,
                    pgvector_query_ms=pgvector_query_ms,
                    returned_chunks=0,
                )
                msg = "No relevant information found in knowledge base."
                if stale_count > 0:
                    msg += (
                        f" ({stale_count} chunk(s) were excluded because they were embedded "
                        f"with a different model — reprocess those documents to include them.)"
                    )
                return ToolResult(ok=True, content=msg)

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
                return ToolResult(ok=True, content="No relevant information found in knowledge base.")

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
                prefix = (chunk.context_prefix or "").strip()
                if prefix:
                    content = f"{prefix} {content}"
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

            content = "\n---\n".join(blocks)
            if stale_count > 0:
                content += (
                    f"\n\n⚠️ Note: {stale_count} chunk(s) in your knowledge base were embedded "
                    f"with a different model and were excluded from this search. "
                    f"Reprocess those documents to make them searchable again."
                )
            return ToolResult(ok=True, content=content)
        except Exception:
            logger.error("rag_search_failed_unexpectedly", exc_info=True)
            return ToolResult(ok=False, content="Retrieval failed unexpectedly.")
