from __future__ import annotations

from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
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


async def get_document_by_filename(
    session: AsyncSession,
    filename: str,
    user_id: str | None = None,
) -> Document | None:
    stmt = select(Document).where(Document.filename == filename)
    if user_id is not None:
        stmt = stmt.where(Document.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalars().first()


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
    query_text: str | None = None,
    top_k: int,
    metadata_filter: dict | None = None,
) -> list[tuple[DocumentChunk, Document, float]]:
    """
    Retrieve relevant chunks using dense vector search or hybrid search.

    When ``settings.hybrid_search_enabled`` is True and ``query_text`` is
    provided, combines cosine similarity (dense) with PostgreSQL full-text
    search (sparse) via Reciprocal Rank Fusion (RRF).  The returned score is
    the RRF score in that case, otherwise it is cosine similarity in [0, 1].

    Returns tuples: (chunk, document, score)
    """
    metadata_filter = metadata_filter or {}

    if settings.hybrid_search_enabled and query_text and query_text.strip():
        return await _hybrid_search(
            session,
            query_vector=query_vector,
            query_text=query_text.strip(),
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    return await _dense_search(
        session,
        query_vector=query_vector,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )


async def _dense_search(
    session: AsyncSession,
    *,
    query_vector: list[float],
    top_k: int,
    metadata_filter: dict,
) -> list[tuple[DocumentChunk, Document, float]]:
    """Pure cosine-similarity search (existing behaviour)."""
    similarity_expr = (1 - DocumentChunk.embedding.cosine_distance(query_vector)).label("similarity")
    stmt = (
        select(DocumentChunk, Document, similarity_expr)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(similarity_expr.desc())
        .limit(top_k)
    )
    stmt = _apply_metadata_filters(stmt, metadata_filter)
    result = await session.execute(stmt)
    return [
        (chunk, doc, float(sim) if sim is not None else 0.0)
        for chunk, doc, sim in result.all()
    ]


async def _hybrid_search(
    session: AsyncSession,
    *,
    query_vector: list[float],
    query_text: str,
    top_k: int,
    metadata_filter: dict,
) -> list[tuple[DocumentChunk, Document, float]]:
    """
    Two-leg hybrid search with Reciprocal Rank Fusion.

    Leg 1 — dense:  cosine distance on pgvector embeddings.
    Leg 2 — sparse: PostgreSQL ts_rank on the generated tsvector column.

    Both legs fetch (top_k * hybrid_fetch_multiplier) candidates so that
    documents scoring well on only one leg are still eligible for the final
    ranking.  RRF score = 1/(k+rank_dense) + 1/(k+rank_sparse).
    """
    rrf_k = settings.hybrid_rrf_k
    fetch_k = max(top_k, top_k * settings.hybrid_fetch_multiplier)

    # --- Dense leg: ranked chunk IDs by cosine similarity ---
    dense_sim_expr = (
        1 - DocumentChunk.embedding.cosine_distance(query_vector)
    ).label("sim")
    dense_stmt = (
        select(DocumentChunk.id, dense_sim_expr)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(dense_sim_expr.desc())
        .limit(fetch_k)
    )
    dense_stmt = _apply_metadata_filters(dense_stmt, metadata_filter)
    dense_result = await session.execute(dense_stmt)
    dense_rows = dense_result.all()
    # {chunk_id: (dense_rank_1based, dense_similarity)}
    dense_rank: dict[int, tuple[int, float]] = {
        row.id: (i + 1, float(row.sim))
        for i, row in enumerate(dense_rows)
    }

    # --- Sparse leg: ranked chunk IDs by ts_rank ---
    sparse_sql = text(
        "SELECT dc.id, ts_rank(dc.text_search, plainto_tsquery('english', :q)) AS fts_score "
        "FROM document_chunks dc "
        "JOIN documents d ON d.id = dc.document_id "
        + _metadata_where_sql(metadata_filter)
        + (" AND " if metadata_filter else " WHERE ")
        + "dc.text_search @@ plainto_tsquery('english', :q) "
        "ORDER BY fts_score DESC "
        "LIMIT :fetch_k"
    )
    sparse_params = {"q": query_text, "fetch_k": fetch_k}
    sparse_params.update(_metadata_sql_params(metadata_filter))
    sparse_result = await session.execute(sparse_sql, sparse_params)
    sparse_rows = sparse_result.all()
    # {chunk_id: sparse_rank_1based}
    sparse_rank: dict[int, int] = {row.id: i + 1 for i, row in enumerate(sparse_rows)}

    # --- RRF fusion ---
    all_ids = set(dense_rank) | set(sparse_rank)
    rrf_scores: list[tuple[int, float]] = []
    for chunk_id in all_ids:
        d_rank = dense_rank.get(chunk_id)
        s_rank = sparse_rank.get(chunk_id)
        score = (1.0 / (rrf_k + d_rank[0]) if d_rank else 0.0) + (
            1.0 / (rrf_k + s_rank) if s_rank else 0.0
        )
        rrf_scores.append((chunk_id, score))

    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    top_ids = [chunk_id for chunk_id, _ in rrf_scores[:top_k]]
    id_to_rrf = {chunk_id: score for chunk_id, score in rrf_scores[:top_k]}

    if not top_ids:
        return []

    # --- Load ORM objects for the final top-k IDs ---
    load_stmt = (
        select(DocumentChunk, Document)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(DocumentChunk.id.in_(top_ids))
    )
    load_result = await session.execute(load_stmt)
    id_to_objects: dict[int, tuple[DocumentChunk, Document]] = {
        chunk.id: (chunk, doc) for chunk, doc in load_result.all()
    }

    # Return in RRF order with the RRF score as the similarity value
    rows: list[tuple[DocumentChunk, Document, float]] = []
    for chunk_id in top_ids:
        if chunk_id in id_to_objects:
            chunk, doc = id_to_objects[chunk_id]
            rows.append((chunk, doc, id_to_rrf[chunk_id]))
    return rows


def _apply_metadata_filters(stmt, metadata_filter: dict):
    """Apply user_id / document_type / tags WHERE clauses to a SQLAlchemy stmt."""
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

    return stmt


def _metadata_where_sql(metadata_filter: dict) -> str:
    """Return raw SQL WHERE fragment (without the WHERE keyword) for metadata filters."""
    clauses: list[str] = []
    user_id = metadata_filter.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        clauses.append("d.user_id = :filter_user_id")
    document_type = metadata_filter.get("document_type")
    if isinstance(document_type, str) and document_type.strip():
        clauses.append("d.document_type = :filter_document_type")
    if not clauses:
        return ""
    return "WHERE " + " AND ".join(clauses)


def _metadata_sql_params(metadata_filter: dict) -> dict:
    """Return bind parameters that match _metadata_where_sql."""
    params: dict = {}
    user_id = metadata_filter.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        params["filter_user_id"] = user_id
    document_type = metadata_filter.get("document_type")
    if isinstance(document_type, str) and document_type.strip():
        params["filter_document_type"] = document_type
    return params


async def count_documents(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Document))
    return int(result.scalar_one() or 0)


async def count_document_chunks(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(DocumentChunk))
    return int(result.scalar_one() or 0)
