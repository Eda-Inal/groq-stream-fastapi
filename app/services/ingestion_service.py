from __future__ import annotations

import time
from dataclasses import replace

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.repositories.document import (
    create_document,
    create_document_chunk,
    delete_document_chunks,
    get_document_by_filename,
    get_document_by_id,
    list_document_chunks,
)
from app.schemas.document import DocumentIngestRequest, DocumentIngestResponse
from app.services.chunking import DocumentTooLargeError, chunk_document
from app.services.embeddings import EmbeddingService

logger = structlog.get_logger()


class DuplicateDocumentError(ValueError):
    """Raised when a document with the same filename (and user_id) already exists."""


class IngestionService:
    def __init__(self) -> None:
        self.embeddings = EmbeddingService()

    async def ingest_document(
        self,
        *,
        session: AsyncSession,
        payload: DocumentIngestRequest,
    ) -> DocumentIngestResponse:
        existing = await get_document_by_filename(
            session,
            filename=payload.filename,
            user_id=payload.user_id,
        )
        if existing is not None:
            raise DuplicateDocumentError(
                f"Document '{payload.filename}' already exists (id={existing.id}). "
                "Delete or reprocess the existing document instead."
            )

        started = time.perf_counter()
        chunks = chunk_document(
            payload.text,
            section_heading=payload.section_heading,
            source_filename=payload.filename,
        )
        if not chunks:
            raise ValueError("Document produced zero chunks after preprocessing.")

        embeddings = await self.embeddings.embed_batch([
            f"{c.context_prefix} {c.text}".strip() if c.context_prefix else c.text
            for c in chunks
        ])
        if embeddings is None:
            raise RuntimeError(
                "Embedding failed; document not persisted. "
                "Please retry the upload. If the problem persists, check the embedding service."
            )

        doc = await create_document(
            session,
            filename=payload.filename,
            source=payload.source,
            document_type=payload.document_type,
            tags=payload.tags,
            user_id=payload.user_id,
            embedding_model_name=embeddings[0].model_name,
            chunk_count=len(chunks),
            conversation_id=payload.conversation_id,
        )

        total = len(chunks)
        chunks = [replace(c, chunk_index=i, total_chunks=total, doc_id=doc.id) for i, c in enumerate(chunks)]

        chunks_created = 0
        tokens_processed = 0

        for item, emb in zip(chunks, embeddings):
            await create_document_chunk(
                session,
                document_id=doc.id,
                chunk_index=item.chunk_index,
                text=item.text,
                embedding=emb.vector,
                chunk_token_count=item.token_count,
                page_number=item.page_number,
                section_heading=item.section_heading,
                context_prefix=item.context_prefix,
                embedding_model_name=settings.embedding_model_name,
            )
            tokens_processed += item.token_count
            chunks_created += 1

        resolved_model_name = embeddings[0].model_name

        await session.commit()
        await session.refresh(doc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "document_ingested",
            document_id=doc.id,
            chunks_created=chunks_created,
            tokens_processed=tokens_processed,
            elapsed_ms=elapsed_ms,
        )

        return DocumentIngestResponse(
            document_id=doc.id,
            chunks_created=chunks_created,
            chunks_skipped=0,
            tokens_processed=tokens_processed,
            elapsed_ms=elapsed_ms,
            embedding_model=resolved_model_name or "unknown",
        )

    async def reprocess_document(
        self,
        *,
        session: AsyncSession,
        document_id: int,
        replacement_text: str | None = None,
        section_heading: str | None = None,
    ) -> DocumentIngestResponse:
        doc = await get_document_by_id(session, document_id)
        if doc is None:
            raise ValueError("Document not found.")

        source_text = replacement_text
        if not source_text:
            old_chunks = await list_document_chunks(session, document_id=document_id)
            source_text = "\n\n".join(c.text for c in old_chunks if c.text.strip())

        if not source_text:
            raise ValueError("Document has no text to reprocess. Provide replacement text.")

        # Remove old chunk vectors and re-ingest on existing document row
        await delete_document_chunks(session, document_id=document_id)
        await session.flush()

        chunks = chunk_document(
            source_text,
            section_heading=section_heading,
            doc_id=document_id,
            source_filename=doc.filename,
        )
        if not chunks:
            raise ValueError("Document produced zero chunks after preprocessing.")

        started = time.perf_counter()
        embeddings = await self.embeddings.embed_batch([
            f"{c.context_prefix} {c.text}".strip() if c.context_prefix else c.text
            for c in chunks
        ])
        if embeddings is None:
            raise RuntimeError(
                "Embedding failed; document not reprocessed. "
                "The existing document record is intact — retry reprocess or check the embedding service."
            )

        total = len(chunks)
        chunks = [replace(c, chunk_index=i, total_chunks=total) for i, c in enumerate(chunks)]

        chunks_created = 0
        tokens_processed = 0
        resolved_model_name: str | None = None

        for item, emb in zip(chunks, embeddings):
            await create_document_chunk(
                session,
                document_id=document_id,
                chunk_index=item.chunk_index,
                text=item.text,
                embedding=emb.vector,
                chunk_token_count=item.token_count,
                page_number=item.page_number,
                section_heading=item.section_heading,
                context_prefix=item.context_prefix,
                embedding_model_name=settings.embedding_model_name,
            )
            tokens_processed += item.token_count
            chunks_created += 1
            resolved_model_name = emb.model_name

        doc.chunk_count = chunks_created
        doc.embedding_model_name = resolved_model_name
        await session.commit()
        await session.refresh(doc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return DocumentIngestResponse(
            document_id=doc.id,
            chunks_created=chunks_created,
            chunks_skipped=0,
            tokens_processed=tokens_processed,
            elapsed_ms=elapsed_ms,
            embedding_model=resolved_model_name or "unknown",
        )

    async def ingest_pdf(
        self,
        *,
        session: AsyncSession,
        content: bytes,
        filename: str,
        user_id: str | None,
        tags: list[str],
        conversation_id: str | None = None,
    ) -> DocumentIngestResponse:
        from app.services.pdf_extractor import PDFExtractionError, extract_pages

        pages = extract_pages(content)

        existing = await get_document_by_filename(session, filename=filename, user_id=user_id)
        if existing is not None:
            raise DuplicateDocumentError(
                f"Document '{filename}' already exists (id={existing.id})."
            )

        doc = await create_document(
            session,
            filename=filename,
            source=None,
            document_type="pdf",
            tags=tags,
            user_id=user_id,
            embedding_model_name=None,
            chunk_count=0,
            conversation_id=conversation_id,
        )

        started = time.perf_counter()

        # Merge all pages into one text so overlap can span page boundaries.
        # Track each page's character range in the merged string to re-assign
        # page_number per chunk after chunking.
        page_char_ranges: list[tuple[int, int, int]] = []
        merged_parts: list[str] = []
        char_pos = 0
        for page in pages:
            text = page["text"]
            page_char_ranges.append((char_pos, char_pos + len(text), page["page"]))
            merged_parts.append(text)
            char_pos += len(text) + 2  # len("\n\n")

        merged_text = "\n\n".join(merged_parts)
        all_chunks = list(chunk_document(merged_text, doc_id=doc.id, source_filename=filename))

        if not all_chunks:
            raise ValueError("PDF produced zero chunks after preprocessing.")

        def _page_for(chunk_text: str) -> int | None:
            # Search for the chunk tail (not head — head may be an overlap prefix
            # from the previous page). rfind returns the last match, which is safe
            # since chunks are assigned in order and later chunks appear later.
            tail = chunk_text.rstrip()[-60:]
            idx = merged_text.rfind(tail)
            if idx == -1:
                tail = chunk_text.rstrip()[-20:]
                idx = merged_text.rfind(tail)
            if idx == -1:
                return None
            for c_start, c_end, page_num in page_char_ranges:
                if c_start <= idx < c_end:
                    return page_num
            return None

        total = len(all_chunks)
        all_chunks = [
            replace(c, chunk_index=i, total_chunks=total, page_number=_page_for(c.text))
            for i, c in enumerate(all_chunks)
        ]

        embeddings = await self.embeddings.embed_batch([
            f"{c.context_prefix} {c.text}".strip() if c.context_prefix else c.text
            for c in all_chunks
        ])
        if embeddings is None:
            raise RuntimeError(
                "Embedding failed; PDF not persisted. "
                "Please retry the upload. If the problem persists, check the embedding service."
            )

        chunks_created = 0
        tokens_processed = 0
        resolved_model_name: str | None = None

        for item, emb in zip(all_chunks, embeddings):
            await create_document_chunk(
                session,
                document_id=doc.id,
                chunk_index=item.chunk_index,
                text=item.text,
                embedding=emb.vector,
                chunk_token_count=item.token_count,
                page_number=item.page_number,
                section_heading=item.section_heading,
                context_prefix=item.context_prefix,
                embedding_model_name=settings.embedding_model_name,
            )
            chunks_created += 1
            tokens_processed += item.token_count
            resolved_model_name = emb.model_name

        doc.chunk_count = chunks_created
        doc.embedding_model_name = resolved_model_name
        await session.commit()
        await session.refresh(doc)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "pdf_ingested",
            document_id=doc.id,
            chunks_created=chunks_created,
            tokens_processed=tokens_processed,
            elapsed_ms=elapsed_ms,
        )

        return DocumentIngestResponse(
            document_id=doc.id,
            chunks_created=chunks_created,
            chunks_skipped=0,
            tokens_processed=tokens_processed,
            elapsed_ms=elapsed_ms,
            embedding_model=resolved_model_name or "unknown",
        )

    @staticmethod
    def validate_ingestion_error(exc: Exception) -> tuple[int, str]:
        from app.services.pdf_extractor import PDFExtractionError

        if isinstance(exc, DuplicateDocumentError):
            return 409, str(exc)
        if isinstance(exc, DocumentTooLargeError):
            return 413, str(exc)
        if isinstance(exc, PDFExtractionError):
            return 422, str(exc)
        if isinstance(exc, ValueError):
            return 400, str(exc)
        return 500, "Document ingestion failed."
