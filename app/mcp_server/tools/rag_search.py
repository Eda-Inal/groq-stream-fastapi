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

logger = structlog.get_logger()


class RagSearchTool(Tool):
    name = "rag_search"
    description = (
        "Search the private knowledge base for relevant document passages. "
        "Use this for uploaded documents, internal policies, and project-specific facts."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {
                "type": "integer",
                "description": "Number of chunks to retrieve (default 5).",
                "default": 5,
            },
            "similarity_threshold": {
                "type": "number",
                "description": "Minimum similarity score to include (default 0.7).",
                "default": 0.7,
            },
            "metadata_filter": {
                "type": "object",
                "description": (
                    "Optional filter object. Supported keys: user_id (string), "
                    "document_type (string), tags (array of strings)."
                ),
            },
        },
        "required": ["query"],
    }

    def __init__(self) -> None:
        self.embeddings = EmbeddingService()

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

            query_started = time.perf_counter()
            async with AsyncSessionLocal() as session:
                rows = await search_document_chunks(
                    session,
                    query_vector=emb.vector,
                    top_k=top_k,
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

            filtered_rows = [(chunk, doc, sim) for chunk, doc, sim in rows if sim >= threshold]
            below_threshold_count = len(rows) - len(filtered_rows)

            top_similarity = max(sim for _, _, sim in rows)
            if top_similarity < threshold:
                rag_metrics.record_retrieval(
                    top_similarity=top_similarity,
                    embedding_latency_ms=embedding_latency_ms,
                    pgvector_query_ms=pgvector_query_ms,
                    returned_chunks=0,
                )
                return "No relevant information found in knowledge base."

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
                            f"Similarity: {similarity:.3f}",
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
