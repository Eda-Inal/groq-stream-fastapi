from __future__ import annotations

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk


async def create_document(
    session: AsyncSession,
    *,
    filename: str,
    source: str | None,
    document_type: str,
    tags: list[str] | None,
    user_id: str | None,
    embedding_model_name: str | None,
    chunk_count: int | None,
) -> Document:
    doc = Document(
        filename=filename,
        source=source,
        document_type=document_type,
        tags=tags,
        user_id=user_id,
        embedding_model_name=embedding_model_name,
        chunk_count=chunk_count,
    )
    session.add(doc)
    await session.flush()
    await session.refresh(doc)
    return doc


async def create_document_chunk(
    session: AsyncSession,
    *,
    document_id: int,
    chunk_index: int,
    text: str,
    embedding: list[float],
    chunk_token_count: int | None,
    page_number: int | None,
    section_heading: str | None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        embedding=embedding,
        chunk_token_count=chunk_token_count,
        page_number=page_number,
        section_heading=section_heading,
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def get_document_by_id(session: AsyncSession, document_id: int) -> Document | None:
    result = await session.execute(select(Document).where(Document.id == document_id))
    return result.scalar_one_or_none()


async def list_documents(
    session: AsyncSession,
    *,
    user_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    if user_id:
        stmt = stmt.where(Document.user_id == user_id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_document_chunks(session: AsyncSession, *, document_id: int) -> list[DocumentChunk]:
    result = await session.execute(
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index.asc())
    )
    return list(result.scalars().all())


async def delete_document_chunks(session: AsyncSession, *, document_id: int) -> None:
    await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))


async def delete_document_by_id(session: AsyncSession, *, document_id: int) -> bool:
    doc = await get_document_by_id(session, document_id)
    if doc is None:
        return False
    await session.delete(doc)
    return True


async def search_document_chunks(
    session: AsyncSession,
    *,
    query_vector: list[float],
    top_k: int,
    metadata_filter: dict | None = None,
) -> list[tuple[DocumentChunk, Document, float]]:
    """
    Vector similarity search with optional metadata filters.

    Returns tuples: (chunk, document, similarity in [0,1]-ish)
    """
    similarity_expr = (1 - DocumentChunk.embedding.cosine_distance(query_vector)).label("similarity")
    stmt = (
        select(DocumentChunk, Document, similarity_expr)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(similarity_expr.desc())
        .limit(top_k)
    )

    metadata_filter = metadata_filter or {}
    user_id = metadata_filter.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        stmt = stmt.where(Document.user_id == user_id)

    document_type = metadata_filter.get("document_type")
    if isinstance(document_type, str) and document_type.strip():
        stmt = stmt.where(Document.document_type == document_type)

    tags = metadata_filter.get("tags")
    if isinstance(tags, list) and tags:
        normalized = [str(t) for t in tags if isinstance(t, (str, int, float))]
        if normalized:
            stmt = stmt.where(Document.tags.contains(normalized))

    result = await session.execute(stmt)
    rows = []
    for chunk, doc, sim in result.all():
        similarity = float(sim) if sim is not None else 0.0
        rows.append((chunk, doc, similarity))
    return rows


async def count_documents(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Document))
    return int(result.scalar_one() or 0)


async def count_document_chunks(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(DocumentChunk))
    return int(result.scalar_one() or 0)
