from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from app.core.config import settings
from app.db.repositories.document import search_document_chunks
from app.db.session import AsyncSessionLocal
from app.mcp_server.tools.base import Tool
from app.services.embeddings import EmbeddingService

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

            emb = await self.embeddings.embed_text(query.strip())
            if emb is None:
                return "Retrieval unavailable: embedding error."

            async with AsyncSessionLocal() as session:
                rows = await search_document_chunks(
                    session,
                    query_vector=emb.vector,
                    top_k=top_k,
                    similarity_threshold=threshold,
                    metadata_filter=metadata_filter,
                )

            if not rows:
                return "No relevant information found in knowledge base."

            top_similarity = max(sim for _, _, sim in rows)
            if top_similarity < threshold:
                return "No relevant information found in knowledge base."

            blocks: list[str] = []
            for chunk, doc, similarity in rows:
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
                chunks_returned=len(rows),
                top_similarity=top_similarity,
                has_metadata_filter=bool(metadata_filter),
            )

            return "\n---\n".join(blocks)
        except Exception:
            logger.error("rag_search_failed_unexpectedly", exc_info=True)
            return "Retrieval failed unexpectedly."
