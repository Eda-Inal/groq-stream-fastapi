from __future__ import annotations

import json as _json
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.document import delete_document_by_id, get_document_by_id, list_documents
from app.db.session import get_db
from app.schemas.document import (
    DocumentIngestRequest,
    DocumentIngestResponse,
    DocumentListResponse,
    DocumentRead,
    DocumentReprocessRequest,
    DocumentUpdateRequest,
)
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["documents"])
logger = structlog.get_logger()
ingestion_service = IngestionService()


def _to_document_read(item) -> DocumentRead:
    created = item.created_at
    if isinstance(created, datetime):
        created_str = created.isoformat()
    else:
        created_str = str(created)

    tags = item.tags if isinstance(item.tags, list) else None
    return DocumentRead(
        id=item.id,
        filename=item.filename,
        source=item.source,
        document_type=item.document_type,
        tags=tags,
        user_id=item.user_id,
        embedding_model_name=item.embedding_model_name,
        chunk_count=item.chunk_count,
        created_at=created_str,
    )


@router.post("/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str | None = Form(default=None),
    tags: str | None = Form(default=None),  # JSON string: '["tag1","tag2"]'
    db: AsyncSession = Depends(get_db),
) -> DocumentIngestResponse:
    filename = file.filename or "upload"
    parsed_tags: list[str] = _json.loads(tags) if tags else []
    content = await file.read()

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported on this endpoint.")

    try:
        return await ingestion_service.ingest_pdf(
            session=db,
            content=content,
            filename=filename,
            user_id=user_id,
            tags=parsed_tags,
        )
    except Exception as exc:
        await db.rollback()
        status, detail = ingestion_service.validate_ingestion_error(exc)
        logger.error("pdf_upload_failed", error=str(exc), status=status, exc_info=True)
        raise HTTPException(status_code=status, detail=detail) from exc


@router.post("", response_model=DocumentIngestResponse)
async def ingest_document(
    payload: DocumentIngestRequest,
    db: AsyncSession = Depends(get_db),
) -> DocumentIngestResponse:
    try:
        return await ingestion_service.ingest_document(session=db, payload=payload)
    except Exception as exc:
        await db.rollback()
        status, detail = ingestion_service.validate_ingestion_error(exc)
        logger.error("document_ingest_failed", error=str(exc), status=status, exc_info=True)
        raise HTTPException(status_code=status, detail=detail) from exc


@router.get("", response_model=DocumentListResponse)
async def get_documents(
    user_id: str | None = Query(default=None),
    tags: list[str] | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    docs = await list_documents(session=db, user_id=user_id, limit=limit, offset=offset)
    if tags:
        wanted = set(tags)
        docs = [
            d
            for d in docs
            if isinstance(d.tags, list) and bool(wanted.intersection(set(str(x) for x in d.tags)))
        ]
    items = [_to_document_read(d) for d in docs]
    return DocumentListResponse(total=len(items), items=items)


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(document_id: int, db: AsyncSession = Depends(get_db)) -> DocumentRead:
    doc = await get_document_by_id(db, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _to_document_read(doc)


@router.put("/{document_id}", response_model=DocumentRead)
async def update_document(
    document_id: int,
    payload: DocumentUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> DocumentRead:
    doc = await get_document_by_id(db, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return _to_document_read(doc)

    for key, value in updates.items():
        setattr(doc, key, value)

    await db.commit()
    await db.refresh(doc)
    return _to_document_read(doc)


@router.delete("/{document_id}")
async def delete_document(document_id: int, db: AsyncSession = Depends(get_db)) -> dict[str, str]:
    ok = await delete_document_by_id(db, document_id=document_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found.")
    await db.commit()
    return {"status": "deleted"}


@router.post("/{document_id}/reprocess", response_model=DocumentIngestResponse)
async def reprocess_document(
    document_id: int,
    payload: DocumentReprocessRequest,
    db: AsyncSession = Depends(get_db),
) -> DocumentIngestResponse:
    try:
        return await ingestion_service.reprocess_document(
            session=db,
            document_id=document_id,
            replacement_text=payload.text,
            section_heading=payload.section_heading,
        )
    except Exception as exc:
        await db.rollback()
        status, detail = ingestion_service.validate_ingestion_error(exc)
        logger.error("document_reprocess_failed", error=str(exc), status=status, exc_info=True)
        raise HTTPException(status_code=status, detail=detail) from exc
