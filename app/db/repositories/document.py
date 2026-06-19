from __future__ import annotations

import re
from sqlalchemy import delete, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk

# Matches structured identifiers: SVC-TXN-0041, ERR::UNMATCHED_RULE,
# dlq://orion-dead-letter-eu, WARN::DEPRECATED_RULESET, INFRA-TKT-20240219-003
_IDENTIFIER_RE = re.compile(
    r'\b[A-Z][A-Z0-9]{1,}(?:[_:\-/]{1,2}[A-Z0-9][A-Z0-9_\-]*)+\b'
)

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "and", "but", "or", "nor", "for", "yet", "so", "in", "on", "at", "to",
    "by", "of", "up", "as", "it", "its", "this", "that", "these", "those",
    "if", "not", "no", "with", "from", "into", "than", "then", "there",
    "their", "they", "them", "about", "after", "before", "between", "any",
    "all", "each", "every", "both", "give", "given", "used", "still",
})

_WORD_RE = re.compile(r'[a-zA-Z0-9]+')


def _build_or_tsquery(query_text: str) -> str | None:
    words = _WORD_RE.findall(query_text)
    terms = [w for w in words if len(w) >= 3 and w.lower() not in _STOP_WORDS]
    if not terms:
        return None
    return " | ".join(terms)


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
    conversation_id: str | None = None,
) -> Document:
    doc = Document(
        filename=filename,
        source=source,
        document_type=document_type,
        tags=tags,
        user_id=user_id,
        embedding_model_name=embedding_model_name,
        chunk_count=chunk_count,
        conversation_id=conversation_id,
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
    context_prefix: str | None = None,
    embedding_model_name: str | None = None,
) -> DocumentChunk:
    chunk = DocumentChunk(
        document_id=document_id,
        chunk_index=chunk_index,
        text=text,
        embedding=embedding,
        chunk_token_count=chunk_token_count,
        page_number=page_number,
        section_heading=section_heading,
        context_prefix=context_prefix,
        embedding_model_name=embedding_model_name,
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
    tags: list[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Document]:
    stmt = select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    if user_id:
        stmt = stmt.where(Document.user_id == user_id)
    if tags:
        normalized = [str(t) for t in tags if isinstance(t, (str, int, float))]
        if normalized:
            stmt = stmt.where(Document.tags.contains(normalized))
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


async def count_stale_chunks(
    session: AsyncSession,
    *,
    user_id: str | None,
    current_model: str,
) -> int:
    """Count chunks whose embedding model differs from current_model (includes NULL)."""
    stmt = (
        select(func.count())
        .select_from(DocumentChunk)
        .join(Document, Document.id == DocumentChunk.document_id)
        .where(
            (DocumentChunk.embedding_model_name == None)  # noqa: E711
            | (DocumentChunk.embedding_model_name != current_model)
        )
    )
    if user_id:
        stmt = stmt.where(Document.user_id == user_id)
    result = await session.execute(stmt)
    return int(result.scalar_one() or 0)


async def search_document_chunks(
    session: AsyncSession,
    *,
    query_vector: list[float],
    query_text: str | None = None,
    top_k: int,
    metadata_filter: dict | None = None,
    embedding_model: str | None = None,
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
            embedding_model=embedding_model,
        )

    return await _dense_search(
        session,
        query_vector=query_vector,
        top_k=top_k,
        metadata_filter=metadata_filter,
        embedding_model=embedding_model,
    )





def _extract_grep_terms(query_text: str) -> list[str]:
    """Extract searchable terms from a natural-language query.

    Two passes:
    1. Identifier extraction — uppercase tokens with separator chars
       (SVC-TXN-0041, ERR::UNMATCHED_RULE, dlq://...). These are searched
       as exact substrings; tsvector destroys them on tokenisation.
    2. Keyword bigrams — consecutive non-stop-word pairs of length ≥ 4.
       Catches natural-language phrases that narrow down the right chunk
       when no identifier appears in the query.

    Returns at most 6 terms to keep the SQL WHERE clause small.
    """
    terms: list[str] = []

    # Pass 1 — identifiers
    for m in _IDENTIFIER_RE.finditer(query_text):
        t = m.group(0)
        if t not in terms:
            terms.append(t)

    # Pass 2 — keyword bigrams (only when identifiers are scarce)
    if len(terms) < 2:
        words = [
            w.lower()
            for w in re.findall(r"[a-zA-Z]+", query_text)
            if len(w) >= 4 and w.lower() not in _STOP_WORDS
        ]
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram not in terms:
                terms.append(bigram)

    return terms[:6]


async def _grep_search(
    session: AsyncSession,
    *,
    query_text: str,
    top_k: int,
    metadata_filter: dict,
    embedding_model: str | None = None,
) -> list[tuple[int, int]]:
    """Exact substring search using pg_trgm ILIKE against raw chunk text.

    Returns [(chunk_id, rank_1based)] sorted by number of term matches
    (chunks matching more terms rank higher). Only chunks matching at
    least one term are returned.

    Requires: pg_trgm extension + GIN index on document_chunks.text
    (migration j5k6l7m8n9o0).
    """
    terms = _extract_grep_terms(query_text)
    if not terms:
        return []

    fetch_k = max(top_k, top_k * settings.hybrid_fetch_multiplier)

    meta_clauses = _metadata_where_clauses(metadata_filter)
    meta_params = _metadata_sql_params(metadata_filter)

    # Build per-term ILIKE expressions that count how many terms each chunk
    # matches. Chunks matching more terms float to the top.
    match_exprs = " + ".join(
        f"(CASE WHEN dc.text ILIKE :grep_term_{i} THEN 1 ELSE 0 END)"
        for i in range(len(terms))
    )
    term_params = {f"grep_term_{i}": f"%{t}%" for i, t in enumerate(terms)}

    where_any = " OR ".join(
        f"dc.text ILIKE :grep_term_{i}" for i in range(len(terms))
    )

    all_where = meta_clauses + [f"({where_any})"]
    if embedding_model:
        all_where.append("dc.embedding_model_name = :embedding_model")
        meta_params["embedding_model"] = embedding_model

    params: dict = {**meta_params, **term_params, "fetch_k": fetch_k}

    sql = text(
        f"SELECT dc.id, ({match_exprs}) AS match_count "
        "FROM document_chunks dc "
        "JOIN documents d ON d.id = dc.document_id "
        "WHERE " + " AND ".join(all_where) + " "
        "ORDER BY match_count DESC, dc.id "
        "LIMIT :fetch_k"
    )

    result = await session.execute(sql, params)
    rows = result.all()
    return [(row.id, i + 1) for i, row in enumerate(rows)]


async def _dense_search(
    session: AsyncSession,
    *,
    query_vector: list[float],
    top_k: int,
    metadata_filter: dict,
    embedding_model: str | None = None,
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
    if embedding_model:
        stmt = stmt.where(DocumentChunk.embedding_model_name == embedding_model)
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
    embedding_model: str | None = None,
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
    if embedding_model:
        dense_stmt = dense_stmt.where(DocumentChunk.embedding_model_name == embedding_model)
    dense_result = await session.execute(dense_stmt)
    dense_rows = dense_result.all()
    # {chunk_id: (dense_rank_1based, dense_similarity)}
    dense_rank: dict[int, tuple[int, float]] = {
        row.id: (i + 1, float(row.sim))
        for i, row in enumerate(dense_rows)
    }

    # --- Sparse leg: ranked chunk IDs by ts_rank ---
    sparse_rank: dict[int, int] = {}
    or_tsquery = _build_or_tsquery(query_text)
    if or_tsquery:
        meta_clauses = _metadata_where_clauses(metadata_filter)
        all_where = meta_clauses + ["dc.text_search @@ to_tsquery('english', :q)"]
        sparse_params: dict = {"q": or_tsquery, "fetch_k": fetch_k}
        sparse_params.update(_metadata_sql_params(metadata_filter))
        if embedding_model:
            all_where.append("dc.embedding_model_name = :embedding_model")
            sparse_params["embedding_model"] = embedding_model
        sparse_sql = text(
            "SELECT dc.id, ts_rank(dc.text_search, to_tsquery('english', :q)) AS fts_score "
            "FROM document_chunks dc "
            "JOIN documents d ON d.id = dc.document_id "
            "WHERE " + " AND ".join(all_where) + " "
            "ORDER BY fts_score DESC "
            "LIMIT :fetch_k"
        )
        sparse_result = await session.execute(sparse_sql, sparse_params)
        sparse_rows = sparse_result.all()
        sparse_rank = {row.id: i + 1 for i, row in enumerate(sparse_rows)}

    # --- Grep leg (third RRF leg) ---
    grep_rank: dict[int, int] = {}
    if settings.grep_search_enabled and query_text:
        grep_rows = await _grep_search(
            session,
            query_text=query_text,
            top_k=fetch_k,
            metadata_filter=metadata_filter,
            embedding_model=embedding_model,
        )
        grep_rank = {chunk_id: rank for chunk_id, rank in grep_rows}

    # --- RRF fusion ---
    all_ids = set(dense_rank) | set(sparse_rank) | set(grep_rank)
    rrf_scores: list[tuple[int, float]] = []
    for chunk_id in all_ids:
        d_rank = dense_rank.get(chunk_id)
        s_rank = sparse_rank.get(chunk_id)
        g_rank = grep_rank.get(chunk_id)
        score = (
            (1.0 / (rrf_k + d_rank[0]) if d_rank else 0.0)
            + (1.0 / (rrf_k + s_rank) if s_rank else 0.0)
            + (1.0 / (rrf_k + g_rank) if g_rank else 0.0)
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


# --- RAG test endpoint (used by rag-test/run_all.py via /tools/rag_debug) ---
# Returns per-leg retrieval breakdown (dense, sparse, grep, RRF fusion)
# with chunk metadata and scores for evaluating retrieval quality.
async def debug_search(
    session: AsyncSession,
    *,
    query_vector: list[float],
    query_text: str | None = None,
    top_k: int,
    metadata_filter: dict | None = None,
    embedding_model: str | None = None,
) -> dict:
    """Like search_document_chunks but returns per-leg breakdown for debugging."""
    metadata_filter = metadata_filter or {}
    rrf_k = settings.hybrid_rrf_k
    fetch_k = max(top_k, top_k * settings.hybrid_fetch_multiplier)

    # --- Dense leg ---
    dense_sim_expr = (
        1 - DocumentChunk.embedding.cosine_distance(query_vector)
    ).label("sim")
    dense_stmt = (
        select(DocumentChunk.id, DocumentChunk.chunk_index, DocumentChunk.document_id, dense_sim_expr)
        .join(Document, Document.id == DocumentChunk.document_id)
        .order_by(dense_sim_expr.desc())
        .limit(fetch_k)
    )
    dense_stmt = _apply_metadata_filters(dense_stmt, metadata_filter)
    if embedding_model:
        dense_stmt = dense_stmt.where(DocumentChunk.embedding_model_name == embedding_model)
    dense_result = await session.execute(dense_stmt)
    dense_rows = dense_result.all()

    dense_leg = []
    dense_rank: dict[int, tuple[int, float]] = {}
    for i, row in enumerate(dense_rows):
        dense_rank[row.id] = (i + 1, float(row.sim))
        dense_leg.append({
            "rank": i + 1,
            "chunk_db_id": row.id,
            "chunk_index": row.chunk_index,
            "document_id": row.document_id,
            "cosine_similarity": round(float(row.sim), 5),
        })

    # --- Sparse leg ---
    sparse_leg = []
    sparse_rank: dict[int, int] = {}
    if settings.hybrid_search_enabled and query_text and query_text.strip():
        or_tsquery = _build_or_tsquery(query_text)
        if or_tsquery:
            meta_clauses = _metadata_where_clauses(metadata_filter)
            all_where = meta_clauses + ["dc.text_search @@ to_tsquery('english', :q)"]
            sparse_params: dict = {"q": or_tsquery, "fetch_k": fetch_k}
            sparse_params.update(_metadata_sql_params(metadata_filter))
            if embedding_model:
                all_where.append("dc.embedding_model_name = :embedding_model")
                sparse_params["embedding_model"] = embedding_model
            sparse_sql = text(
                "SELECT dc.id, dc.chunk_index, dc.document_id, "
                "ts_rank(dc.text_search, to_tsquery('english', :q)) AS fts_score "
                "FROM document_chunks dc "
                "JOIN documents d ON d.id = dc.document_id "
                "WHERE " + " AND ".join(all_where) + " "
                "ORDER BY fts_score DESC "
                "LIMIT :fetch_k"
            )
            sparse_result = await session.execute(sparse_sql, sparse_params)
            for i, row in enumerate(sparse_result.all()):
                sparse_rank[row.id] = i + 1
                sparse_leg.append({
                    "rank": i + 1,
                    "chunk_db_id": row.id,
                    "chunk_index": row.chunk_index,
                    "document_id": row.document_id,
                    "fts_score": round(float(row.fts_score), 5),
                })

    # --- Grep leg ---
    grep_leg = []
    grep_rank: dict[int, int] = {}
    if settings.grep_search_enabled and query_text:
        grep_rows = await _grep_search(
            session,
            query_text=query_text,
            top_k=fetch_k,
            metadata_filter=metadata_filter,
            embedding_model=embedding_model,
        )
        grep_rank = {chunk_id: rank for chunk_id, rank in grep_rows}
        for chunk_id, rank in grep_rows:
            grep_leg.append({"rank": rank, "chunk_db_id": chunk_id})

    # --- RRF fusion ---
    all_ids = set(dense_rank) | set(sparse_rank) | set(grep_rank)
    rrf_scores: list[tuple[int, float, float, float, float]] = []
    for chunk_id in all_ids:
        d = dense_rank.get(chunk_id)
        s = sparse_rank.get(chunk_id)
        g = grep_rank.get(chunk_id)
        d_contrib = 1.0 / (rrf_k + d[0]) if d else 0.0
        s_contrib = 1.0 / (rrf_k + s) if s else 0.0
        g_contrib = 1.0 / (rrf_k + g) if g else 0.0
        total = d_contrib + s_contrib + g_contrib
        rrf_scores.append((chunk_id, total, d_contrib, s_contrib, g_contrib))

    rrf_scores.sort(key=lambda x: x[1], reverse=True)
    top_ids = [cid for cid, _, _, _, _ in rrf_scores[:top_k]]

    # --- Load full chunk objects ---
    if top_ids:
        load_stmt = (
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(DocumentChunk.id.in_(top_ids))
        )
        load_result = await session.execute(load_stmt)
        id_to_obj = {chunk.id: (chunk, doc) for chunk, doc in load_result.all()}
    else:
        id_to_obj = {}

    pre_rerank = []
    for rank, (cid, total, d_c, s_c, g_c) in enumerate(rrf_scores[:top_k], 1):
        entry = {
            "rank": rank,
            "chunk_db_id": cid,
            "rrf_score": round(total, 6),
            "dense_contrib": round(d_c, 6),
            "sparse_contrib": round(s_c, 6),
            "grep_contrib": round(g_c, 6),
        }
        if cid in dense_rank:
            entry["cosine_similarity"] = round(dense_rank[cid][1], 5)
        obj = id_to_obj.get(cid)
        if obj:
            chunk, doc = obj
            entry["chunk_index"] = chunk.chunk_index
            entry["document_id"] = chunk.document_id
            entry["filename"] = doc.filename
            entry["section_heading"] = chunk.section_heading
            entry["page_number"] = chunk.page_number
            entry["chunk_token_count"] = chunk.chunk_token_count
            entry["text_preview"] = chunk.text[:200].strip()
            entry["text_full"] = chunk.text.strip()
        pre_rerank.append(entry)

    return {
        "search_mode": "hybrid" if (settings.hybrid_search_enabled and query_text) else "dense",
        "rrf_k": rrf_k,
        "fetch_k": fetch_k,
        "requested_top_k": top_k,
        "dense_candidates": len(dense_leg),
        "sparse_candidates": len(sparse_leg),
        "grep_candidates": len(grep_leg),
        "dense_leg": dense_leg[:20],
        "sparse_leg": sparse_leg[:20],
        "grep_leg": grep_leg[:20],
        "pre_rerank": pre_rerank,
    }


def _apply_metadata_filters(stmt, metadata_filter: dict):
    """Apply user_id / document_type / tags / conversation_id WHERE clauses to a SQLAlchemy stmt."""
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

    conversation_id = metadata_filter.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        stmt = stmt.where(Document.conversation_id == conversation_id)

    return stmt


def _metadata_where_clauses(metadata_filter: dict) -> list[str]:
    """Return a list of raw SQL boolean expressions for metadata filters."""
    clauses: list[str] = []
    user_id = metadata_filter.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        clauses.append("d.user_id = :filter_user_id")
    document_type = metadata_filter.get("document_type")
    if isinstance(document_type, str) and document_type.strip():
        clauses.append("d.document_type = :filter_document_type")
    tags = metadata_filter.get("tags")
    if isinstance(tags, list) and tags:
        clauses.append("d.tags @> :filter_tags::jsonb")
    conversation_id = metadata_filter.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        clauses.append("d.conversation_id = :filter_conversation_id")
    return clauses


def _metadata_sql_params(metadata_filter: dict) -> dict:
    """Return bind parameters that match _metadata_where_clauses."""
    params: dict = {}
    user_id = metadata_filter.get("user_id")
    if isinstance(user_id, str) and user_id.strip():
        params["filter_user_id"] = user_id
    document_type = metadata_filter.get("document_type")
    if isinstance(document_type, str) and document_type.strip():
        params["filter_document_type"] = document_type
    tags = metadata_filter.get("tags")
    if isinstance(tags, list) and tags:
        import json as _json
        params["filter_tags"] = _json.dumps([str(t) for t in tags])
    conversation_id = metadata_filter.get("conversation_id")
    if isinstance(conversation_id, str) and conversation_id.strip():
        params["filter_conversation_id"] = conversation_id
    return params


async def has_documents_for_conversation(
    session: AsyncSession,
    *,
    conversation_id: str,
    user_id: str | None = None,
) -> bool:
    stmt = select(Document.id).where(Document.conversation_id == conversation_id).limit(1)
    if user_id:
        stmt = stmt.where(Document.user_id == user_id)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


# TEST MODE: used when conversation_id injection is disabled.
# Checks whether any documents exist for the given user_id, ignoring conversation_id.
async def has_documents_for_user(
    session: AsyncSession,
    *,
    user_id: str,
) -> bool:
    stmt = select(Document.id).where(Document.user_id == user_id).limit(1)
    result = await session.execute(stmt)
    return result.scalar_one_or_none() is not None


async def count_documents(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(Document))
    return int(result.scalar_one() or 0)


async def count_document_chunks(session: AsyncSession) -> int:
    result = await session.execute(select(func.count()).select_from(DocumentChunk))
    return int(result.scalar_one() or 0)
