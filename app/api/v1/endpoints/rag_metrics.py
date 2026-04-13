from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.document import count_document_chunks, count_documents
from app.db.session import get_db
from app.services.rag_metrics import rag_metrics

router = APIRouter(prefix="/rag", tags=["rag"])


@router.get("/metrics")
async def get_rag_metrics(db: AsyncSession = Depends(get_db)) -> dict[str, float | int]:
    total_documents = await count_documents(db)
    total_chunks = await count_document_chunks(db)
    metrics = rag_metrics.snapshot()
    return {
        "total_documents": total_documents,
        "total_chunks": total_chunks,
        **metrics,
    }
